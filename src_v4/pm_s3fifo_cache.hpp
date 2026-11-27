#pragma once
// Unified Lock-Sharded 4 KiB S3-FIFO Page Cache (Theorem 5 KKT Optimal)
// - Caches 4 KiB NVMe pages across BOTH `tier1_b4.nvme` and `tier2_fp16.nvme` inside ONE pool.
// - Supports BOTH process-local aligned heap allocation AND caller-provided shared memory
//   (e.g., PostgreSQL ShmemInitStruct) with zero pointers inside the shared region!
// - Strictly O_DIRECT backed: OS page cache never silently bloats RSS beyond cgroup MemoryMax.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

namespace pm_certvec {

constexpr size_t kPageSize = 4096;
constexpr uint32_t kCacheNumShards = 64; // Reduces lock contention across 48 cores / 48 PG backends
constexpr uint64_t kCacheShmemMagic = 0x504D53334649464FULL; // "PMS3FIFO"

// Key packing: bits [63:48] = file_id (seg/field/tier), bits [47:0] = 4 KiB page index
static inline uint64_t make_page_key(uint16_t file_id, uint64_t page_idx) {
    return (static_cast<uint64_t>(file_id) << 48) | (page_idx & 0x0000FFFFFFFFFFFFULL);
}

static inline uint64_t splitmix64_hash(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Per-shard lock using process-shared atomic spinlock (works across threads AND PG processes!)
struct alignas(64) S3ShardHeader {
    std::atomic<uint32_t> lock_word;
    uint32_t capacity;       // total 4 KiB slots in this shard
    uint32_t small_cap;      // S3-FIFO Small queue (~10% of capacity)
    uint32_t main_cap;       // S3-FIFO Main queue (~90% of capacity)
    uint32_t ghost_cap;      // S3-FIFO Ghost ring (~90% of capacity)
    uint32_t small_head, small_tail, small_count;
    uint32_t main_head, main_tail, main_count;
    uint32_t ghost_pos;
    uint32_t free_top;
    uint32_t hash_mask;      // power-of-2 - 1 (table size >= 2 * capacity)
    uint64_t hits_b4;
    uint64_t hits_t2;
    uint64_t misses;
    uint64_t evictions;

    inline void lock() noexcept {
        while (lock_word.exchange(1u, std::memory_order_acquire) != 0u) {
            while (lock_word.load(std::memory_order_relaxed) != 0u) {
#if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
#endif
            }
        }
    }
    inline void unlock() noexcept {
        lock_word.store(0u, std::memory_order_release);
    }
};

struct S3SlotMeta {
    uint64_t key;
    uint8_t  freq;   // S3-FIFO 2-bit saturating frequency (0..3)
    uint8_t  in_main;// 0 = Small, 1 = Main, 2 = Free
    uint8_t  tier_tag; // 0 = B4, 1 = T2
    uint8_t  pad;
    uint32_t hash_bucket;
};

struct S3GlobalHeader {
    uint64_t magic;
    uint32_t num_shards;
    uint32_t slots_per_shard;
    uint32_t hash_size_per_shard;
    uint32_t ghost_cap_per_shard;
    uint64_t total_bytes;
    uint64_t shard_meta_off;
    uint64_t slot_meta_off;
    uint64_t hash_tbl_off;
    uint64_t queues_off;
    uint64_t ghost_off;
    uint64_t pages_off; // 4096-byte aligned data pool
};

static inline uint32_t next_pow2_u32(uint32_t v) {
    uint32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

static inline size_t align_up_sz(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

static inline size_t compute_s3fifo_layout(size_t data_bytes, S3GlobalHeader& hdr) {
    uint32_t total_pages = static_cast<uint32_t>(data_bytes / kPageSize);
    uint32_t slots_per = total_pages / kCacheNumShards;
    if (slots_per < 16) slots_per = 16;
    uint32_t hsz = next_pow2_u32(slots_per * 2u + 8u);
    uint32_t gcap = slots_per;

    size_t off = align_up_sz(sizeof(S3GlobalHeader), 64);
    hdr.magic = kCacheShmemMagic;
    hdr.num_shards = kCacheNumShards;
    hdr.slots_per_shard = slots_per;
    hdr.hash_size_per_shard = hsz;
    hdr.ghost_cap_per_shard = gcap;

    hdr.shard_meta_off = off;
    off = align_up_sz(off + static_cast<size_t>(kCacheNumShards) * sizeof(S3ShardHeader), 64);

    hdr.slot_meta_off = off;
    off = align_up_sz(off + static_cast<size_t>(kCacheNumShards) * slots_per * sizeof(S3SlotMeta), 64);

    hdr.hash_tbl_off = off;
    off = align_up_sz(off + static_cast<size_t>(kCacheNumShards) * hsz * sizeof(uint32_t), 64);

    // Per shard: small_ring[slots_per] + main_ring[slots_per] + free_stack[slots_per]
    hdr.queues_off = off;
    off = align_up_sz(off + static_cast<size_t>(kCacheNumShards) * slots_per * 3ULL * sizeof(uint32_t), 64);

    hdr.ghost_off = off;
    off = align_up_sz(off + static_cast<size_t>(kCacheNumShards) * gcap * sizeof(uint64_t), 64);

    hdr.pages_off = align_up_sz(off, kPageSize);
    hdr.total_bytes = hdr.pages_off + static_cast<size_t>(kCacheNumShards) * slots_per * kPageSize;
    return hdr.total_bytes;
}

class UnifiedS3FifoCache {
public:
    UnifiedS3FifoCache() = default;
    ~UnifiedS3FifoCache() { destroy(); }

    UnifiedS3FifoCache(const UnifiedS3FifoCache&) = delete;
    UnifiedS3FifoCache& operator=(const UnifiedS3FifoCache&) = delete;

    bool init_heap(size_t target_data_bytes) {
        destroy();
        if (target_data_bytes < kCacheNumShards * 16ULL * kPageSize) return false;
        S3GlobalHeader tmp{};
        size_t need = compute_s3fifo_layout(target_data_bytes, tmp);
        void* mem = std::aligned_alloc(kPageSize, align_up_sz(need, kPageSize));
        if (!mem) return false;
        owned_ = true;
        return attach_region(mem, need, true);
    }

    // Attach to caller-provided shared memory (e.g., PostgreSQL ShmemInitStruct).
    // If `force_init == false` and magic already matches, attaches cleanly without wiping!
    bool attach_region(void* region, size_t region_bytes, bool force_init) {
        if (!region || region_bytes < 4096) return false;
        auto* existing = reinterpret_cast<S3GlobalHeader*>(region);
        if (!force_init && existing->magic == kCacheShmemMagic && existing->total_bytes <= region_bytes) {
            base_ = static_cast<uint8_t*>(region);
            hdr_ = existing;
            active_ = true;
            return true;
        }
        // Compute largest slots_per_shard fitting into region_bytes
        size_t lo = kCacheNumShards * 16ULL * kPageSize;
        size_t hi = region_bytes;
        S3GlobalHeader best{};
        size_t best_need = 0;
        if (compute_s3fifo_layout(lo, best) > region_bytes) return false;
        best_need = best.total_bytes;
        while (lo + kPageSize * kCacheNumShards <= hi) {
            size_t mid = lo + ((hi - lo) / 2);
            S3GlobalHeader cand{};
            size_t need = compute_s3fifo_layout(mid, cand);
            if (need <= region_bytes) { best = cand; best_need = need; lo = mid + kPageSize; }
            else { hi = mid - kPageSize; }
        }
        std::memset(region, 0, best_need);
        *existing = best;
        base_ = static_cast<uint8_t*>(region);
        hdr_ = existing;

        for (uint32_t s = 0; s < hdr_->num_shards; ++s) {
            auto* sh = shard_hdr(s);
            sh->lock_word.store(0u, std::memory_order_relaxed);
            sh->capacity = hdr_->slots_per_shard;
            sh->small_cap = std::max<uint32_t>(2u, sh->capacity / 10u);
            sh->main_cap = sh->capacity - sh->small_cap;
            sh->ghost_cap = hdr_->ghost_cap_per_shard;
            sh->hash_mask = hdr_->hash_size_per_shard - 1u;
            sh->free_top = sh->capacity;
            uint32_t* ht = shard_hash(s);
            for (uint32_t i = 0; i < hdr_->hash_size_per_shard; ++i) ht[i] = kEmptySlot;
            uint32_t* free_stk = shard_free(s);
            S3SlotMeta* metas = shard_slots(s);
            for (uint32_t i = 0; i < sh->capacity; ++i) {
                free_stk[i] = sh->capacity - 1u - i;
                metas[i].in_main = 2;
                metas[i].key = kInvalidKey;
            }
            uint64_t* gh = shard_ghost(s);
            for (uint32_t i = 0; i < sh->ghost_cap; ++i) gh[i] = kInvalidKey;
        }
        active_ = true;
        return true;
    }

    void destroy() {
        if (owned_ && base_) std::free(base_);
        base_ = nullptr;
        hdr_ = nullptr;
        owned_ = false;
        active_ = false;
    }

    bool enabled() const noexcept { return active_; }
    size_t data_capacity_bytes() const noexcept {
        return active_ ? static_cast<size_t>(hdr_->num_shards) * hdr_->slots_per_shard * kPageSize : 0;
    }
    size_t total_ram_bytes() const noexcept { return active_ ? hdr_->total_bytes : 0; }

    // Lookup a 4 KiB page; copies into dst_4k on hit. Thread- and multi-process-safe.
    bool get_page(uint16_t file_id, uint64_t page_idx, uint8_t tier_tag, uint8_t* dst_4k) noexcept {
        if (!active_) return false;
        uint64_t key = make_page_key(file_id, page_idx);
        uint64_t h = splitmix64_hash(key);
        uint32_t s = static_cast<uint32_t>(h & (kCacheNumShards - 1u));
        auto* sh = shard_hdr(s);
        sh->lock();
        uint32_t slot = find_slot_locked(s, key, h);
        if (slot != kEmptySlot) {
            S3SlotMeta& m = shard_slots(s)[slot];
            if (m.freq < 3) m.freq++;
            std::memcpy(dst_4k, shard_page(s, slot), kPageSize);
            if (tier_tag == 0) sh->hits_b4++; else sh->hits_t2++;
            sh->unlock();
            return true;
        }
        sh->misses++;
        sh->unlock();
        return false;
    }

    // Insert a 4 KiB page into the unified S3-FIFO cache (following S3-FIFO Small -> Ghost -> Main rules).
    void put_page(uint16_t file_id, uint64_t page_idx, uint8_t tier_tag, const uint8_t* src_4k) noexcept {
        if (!active_) return;
        uint64_t key = make_page_key(file_id, page_idx);
        uint64_t h = splitmix64_hash(key);
        uint32_t s = static_cast<uint32_t>(h & (kCacheNumShards - 1u));
        auto* sh = shard_hdr(s);
        sh->lock();
        if (find_slot_locked(s, key, h) != kEmptySlot) {
            sh->unlock();
            return;
        }
        bool in_ghost = ghost_contains_locked(s, key);
        uint32_t slot = alloc_slot_locked(s, in_ghost);
        if (slot == kEmptySlot) { sh->unlock(); return; }
        S3SlotMeta& m = shard_slots(s)[slot];
        m.key = key;
        m.freq = 0;
        m.tier_tag = tier_tag;
        std::memcpy(shard_page(s, slot), src_4k, kPageSize);
        insert_hash_locked(s, slot, h);
        if (in_ghost && sh->main_count < sh->main_cap) {
            m.in_main = 1;
            push_main_locked(s, slot);
        } else {
            m.in_main = 0;
            push_small_locked(s, slot);
        }
        sh->unlock();
    }

    void reset_stats() noexcept {
        if (!active_) return;
        for (uint32_t s = 0; s < hdr_->num_shards; ++s) {
            auto* sh = shard_hdr(s);
            sh->lock();
            sh->hits_b4 = sh->hits_t2 = sh->misses = sh->evictions = 0;
            sh->unlock();
        }
    }

    void get_stats(uint64_t& hits_b4, uint64_t& hits_t2, uint64_t& misses, uint64_t& evictions) const noexcept {
        hits_b4 = hits_t2 = misses = evictions = 0;
        if (!active_) return;
        for (uint32_t s = 0; s < hdr_->num_shards; ++s) {
            const auto* sh = shard_hdr(s);
            hits_b4 += sh->hits_b4;
            hits_t2 += sh->hits_t2;
            misses += sh->misses;
            evictions += sh->evictions;
        }
    }

private:
    static constexpr uint32_t kEmptySlot  = 0xFFFFFFFFu;
    static constexpr uint64_t kInvalidKey = 0xFFFFFFFFFFFFFFFFULL;

    uint8_t*        base_ = nullptr;
    S3GlobalHeader* hdr_ = nullptr;
    bool            owned_ = false;
    bool            active_ = false;

    S3ShardHeader* shard_hdr(uint32_t s) const noexcept {
        return reinterpret_cast<S3ShardHeader*>(base_ + hdr_->shard_meta_off) + s;
    }
    S3SlotMeta* shard_slots(uint32_t s) const noexcept {
        return reinterpret_cast<S3SlotMeta*>(base_ + hdr_->slot_meta_off) + static_cast<size_t>(s) * hdr_->slots_per_shard;
    }
    uint32_t* shard_hash(uint32_t s) const noexcept {
        return reinterpret_cast<uint32_t*>(base_ + hdr_->hash_tbl_off) + static_cast<size_t>(s) * hdr_->hash_size_per_shard;
    }
    uint32_t* shard_small(uint32_t s) const noexcept {
        return reinterpret_cast<uint32_t*>(base_ + hdr_->queues_off) + static_cast<size_t>(s) * hdr_->slots_per_shard * 3ULL;
    }
    uint32_t* shard_main(uint32_t s) const noexcept { return shard_small(s) + hdr_->slots_per_shard; }
    uint32_t* shard_free(uint32_t s) const noexcept { return shard_small(s) + 2ULL * hdr_->slots_per_shard; }
    uint64_t* shard_ghost(uint32_t s) const noexcept {
        return reinterpret_cast<uint64_t*>(base_ + hdr_->ghost_off) + static_cast<size_t>(s) * hdr_->ghost_cap_per_shard;
    }
    uint8_t* shard_page(uint32_t s, uint32_t slot) const noexcept {
        return base_ + hdr_->pages_off + (static_cast<size_t>(s) * hdr_->slots_per_shard + slot) * kPageSize;
    }

    uint32_t find_slot_locked(uint32_t s, uint64_t key, uint64_t h) const noexcept {
        const auto* sh = shard_hdr(s);
        const uint32_t* ht = shard_hash(s);
        const S3SlotMeta* metas = shard_slots(s);
        uint32_t idx = static_cast<uint32_t>(h >> 6) & sh->hash_mask;
        for (uint32_t probe = 0; probe <= sh->hash_mask; ++probe) {
            uint32_t sl = ht[idx];
            if (sl == kEmptySlot) return kEmptySlot;
            if (metas[sl].key == key) return sl;
            idx = (idx + 1u) & sh->hash_mask;
        }
        return kEmptySlot;
    }

    void insert_hash_locked(uint32_t s, uint32_t slot, uint64_t h) noexcept {
        const auto* sh = shard_hdr(s);
        uint32_t* ht = shard_hash(s);
        uint32_t idx = static_cast<uint32_t>(h >> 6) & sh->hash_mask;
        while (ht[idx] != kEmptySlot) idx = (idx + 1u) & sh->hash_mask;
        ht[idx] = slot;
        shard_slots(s)[slot].hash_bucket = idx;
    }

    void remove_hash_locked(uint32_t s, uint32_t slot) noexcept {
        const auto* sh = shard_hdr(s);
        uint32_t* ht = shard_hash(s);
        S3SlotMeta* metas = shard_slots(s);
        uint32_t mask = sh->hash_mask;
        uint32_t hole = metas[slot].hash_bucket;
        ht[hole] = kEmptySlot;
        // Backward-shift deletion preserves linear probing with zero tombstones!
        uint32_t cur = (hole + 1u) & mask;
        while (ht[cur] != kEmptySlot) {
            uint32_t sl = ht[cur];
            uint32_t ideal = static_cast<uint32_t>(splitmix64_hash(metas[sl].key) >> 6) & mask;
            if (((cur - ideal) & mask) >= ((cur - hole) & mask)) {
                ht[hole] = sl;
                metas[sl].hash_bucket = hole;
                ht[cur] = kEmptySlot;
                hole = cur;
            }
            cur = (cur + 1u) & mask;
        }
    }

    void push_small_locked(uint32_t s, uint32_t slot) noexcept {
        auto* sh = shard_hdr(s);
        shard_small(s)[sh->small_tail] = slot;
        sh->small_tail = (sh->small_tail + 1u) % sh->capacity;
        sh->small_count++;
    }
    uint32_t pop_small_locked(uint32_t s) noexcept {
        auto* sh = shard_hdr(s);
        uint32_t sl = shard_small(s)[sh->small_head];
        sh->small_head = (sh->small_head + 1u) % sh->capacity;
        sh->small_count--;
        return sl;
    }
    void push_main_locked(uint32_t s, uint32_t slot) noexcept {
        auto* sh = shard_hdr(s);
        shard_main(s)[sh->main_tail] = slot;
        sh->main_tail = (sh->main_tail + 1u) % sh->capacity;
        sh->main_count++;
    }
    uint32_t pop_main_locked(uint32_t s) noexcept {
        auto* sh = shard_hdr(s);
        uint32_t sl = shard_main(s)[sh->main_head];
        sh->main_head = (sh->main_head + 1u) % sh->capacity;
        sh->main_count--;
        return sl;
    }

    void ghost_insert_locked(uint32_t s, uint64_t key) noexcept {
        auto* sh = shard_hdr(s);
        uint64_t* gh = shard_ghost(s);
        gh[sh->ghost_pos] = key;
        sh->ghost_pos = (sh->ghost_pos + 1u) % sh->ghost_cap;
    }
    bool ghost_contains_locked(uint32_t s, uint64_t key) const noexcept {
        const auto* sh = shard_hdr(s);
        const uint64_t* gh = shard_ghost(s);
        // Compact per-shard ghost ring (typical ~100-400 entries) — scan is fast & L1-resident;
        // for larger shards cap scan window to most recent 512 ghost entries for O(1) latency.
        uint32_t scan = std::min<uint32_t>(sh->ghost_cap, 512u);
        uint32_t pos = sh->ghost_pos;
        for (uint32_t i = 0; i < scan; ++i) {
            pos = (pos == 0) ? (sh->ghost_cap - 1u) : (pos - 1u);
            if (gh[pos] == key) return true;
        }
        return false;
    }

    uint32_t evict_main_locked(uint32_t s) noexcept {
        auto* sh = shard_hdr(s);
        S3SlotMeta* metas = shard_slots(s);
        while (sh->main_count > 0) {
            uint32_t sl = pop_main_locked(s);
            if (metas[sl].freq > 0) {
                metas[sl].freq--;
                push_main_locked(s, sl);
            } else {
                remove_hash_locked(s, sl);
                sh->evictions++;
                metas[sl].key = kInvalidKey;
                metas[sl].in_main = 2;
                return sl;
            }
        }
        return kEmptySlot;
    }

    uint32_t evict_small_locked(uint32_t s) noexcept {
        auto* sh = shard_hdr(s);
        S3SlotMeta* metas = shard_slots(s);
        while (sh->small_count > 0) {
            uint32_t sl = pop_small_locked(s);
            if (metas[sl].freq > 0) {
                metas[sl].freq = 0;
                metas[sl].in_main = 1;
                push_main_locked(s, sl);
                if (sh->main_count > sh->main_cap) {
                    uint32_t vic = evict_main_locked(s);
                    if (vic != kEmptySlot) return vic;
                }
            } else {
                uint64_t old_k = metas[sl].key;
                remove_hash_locked(s, sl);
                ghost_insert_locked(s, old_k);
                sh->evictions++;
                metas[sl].key = kInvalidKey;
                metas[sl].in_main = 2;
                return sl;
            }
        }
        return evict_main_locked(s);
    }

    uint32_t alloc_slot_locked(uint32_t s, bool want_main) noexcept {
        auto* sh = shard_hdr(s);
        if (sh->free_top > 0) {
            return shard_free(s)[--sh->free_top];
        }
        if (want_main && sh->main_count >= sh->main_cap) {
            uint32_t vic = evict_main_locked(s);
            if (vic != kEmptySlot) return vic;
        }
        if (sh->small_count >= sh->small_cap) {
            uint32_t vic = evict_small_locked(s);
            if (vic != kEmptySlot) return vic;
        }
        return evict_main_locked(s);
    }
};

} // namespace pm_certvec
