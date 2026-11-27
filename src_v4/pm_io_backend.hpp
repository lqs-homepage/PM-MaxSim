#pragma once
// Dual Storage I/O Backend (io_uring + Portable POSIX pread Fallback)
// Seamlessly integrates with UnifiedS3FifoCache so every 4 KiB page miss performs exact O_DIRECT read
// and inserts back into the unified S3-FIFO cache (covering BOTH tier1_b4.nvme and tier2_fp16.nvme!).

#include "pm_certvec.h"
#include "pm_s3fifo_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#ifdef PM_HAS_LIBURING
#include <liburing.h>
#endif

namespace pm_certvec {

struct DocDiskEntry {
    uint64_t offset;
    uint32_t aligned_bytes;
    uint32_t n_toks;
};

template <typename T>
struct AlignedBuf {
    T*     ptr = nullptr;
    size_t sz  = 0;
    AlignedBuf() = default;
    ~AlignedBuf() { if (ptr) std::free(ptr); }
    AlignedBuf(const AlignedBuf&) = delete;
    AlignedBuf& operator=(const AlignedBuf&) = delete;
    AlignedBuf(AlignedBuf&& o) noexcept : ptr(o.ptr), sz(o.sz) { o.ptr = nullptr; o.sz = 0; }
    AlignedBuf& operator=(AlignedBuf&& o) noexcept {
        if (this != &o) { if (ptr) std::free(ptr); ptr = o.ptr; sz = o.sz; o.ptr = nullptr; o.sz = 0; }
        return *this;
    }
    void resize(size_t n, size_t align = 64) {
        if (ptr) std::free(ptr);
        sz = n;
        size_t bytes = ((n * sizeof(T) + align - 1) / align) * align;
        if (bytes == 0) bytes = align;
        ptr = static_cast<T*>(std::aligned_alloc(align, bytes));
        if (!ptr) { std::perror("aligned_alloc"); std::abort(); }
        std::memset(ptr, 0, bytes);
    }
    T* data() { return ptr; }
    const T* data() const { return ptr; }
    size_t size() const { return sz; }
    T& operator[](size_t i) { return ptr[i]; }
    const T& operator[](size_t i) const { return ptr[i]; }
};

struct MappedReadFile {
    int      fd = -1;
    uint8_t* ptr = nullptr;
    size_t   len = 0;

    bool open_mmap(const std::string& path) {
        close_mmap();
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st{};
        if (fstat(fd, &st) != 0 || st.st_size <= 0) { ::close(fd); fd = -1; return false; }
        len = static_cast<size_t>(st.st_size);
        void* p = ::mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) { ::close(fd); fd = -1; len = 0; return false; }
        ptr = static_cast<uint8_t*>(p);
        return true;
    }
    void close_mmap() {
        if (ptr) ::munmap(ptr, len);
        if (fd >= 0) ::close(fd);
        ptr = nullptr; fd = -1; len = 0;
    }
    ~MappedReadFile() { close_mmap(); }
};

struct SharedDiskStore {
    int            fd = -1;
    uint16_t       file_id = 0;
    uint8_t        tier_tag = 0; // 0 = B4, 1 = T2
    bool           in_ram = false;
    bool           o_direct = true;
    uint64_t       file_bytes = 0;
    MappedReadFile meta_map;
    MappedReadFile ram_map;
    const DocDiskEntry* meta = nullptr;
    size_t         n_docs = 0;

    bool open_store(const std::string& path, const std::string& meta_path,
                    uint16_t fid, uint8_t ttag, bool map_ram, bool use_o_direct) {
        file_id = fid;
        tier_tag = ttag;
        o_direct = use_o_direct;
        if (!meta_map.open_mmap(meta_path)) return false;
        meta = reinterpret_cast<const DocDiskEntry*>(meta_map.ptr);
        n_docs = meta_map.len / sizeof(DocDiskEntry);

        struct stat st{};
        if (stat(path.c_str(), &st) != 0) return false;
        file_bytes = static_cast<uint64_t>(st.st_size);

        if (map_ram) {
            if (!ram_map.open_mmap(path)) return false;
            in_ram = true;
            return true;
        }
        int flags = O_RDONLY;
#ifdef O_DIRECT
        if (use_o_direct) flags |= O_DIRECT;
#endif
        fd = ::open(path.c_str(), flags);
        if (fd < 0 && use_o_direct) {
            o_direct = false;
            fd = ::open(path.c_str(), O_RDONLY);
        }
        if (fd < 0) return false;
#ifdef POSIX_FADV_DONTNEED
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
        return true;
    }

    void flush_cold() const {
#ifdef POSIX_FADV_DONTNEED
        if (fd >= 0) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
    }

    ~SharedDiskStore() { if (fd >= 0) ::close(fd); }
};

class ThreadPageReader {
public:
    ThreadPageReader() = default;
    ~ThreadPageReader() { shutdown(); }

    void init(pm_io_backend_t backend, uint32_t qd = 64, size_t init_bounce_kb = 256) {
        shutdown();
        qd_ = std::max<uint32_t>(1u, std::min<uint32_t>(qd, 128u));
        bounce_.resize(init_bounce_kb * 1024ULL, kPageSize);
        backend_ = PM_IO_PREAD_POOL;
#ifdef PM_HAS_LIBURING
        if (backend == PM_IO_URING) {
            if (io_uring_queue_init(128, &ring_, 0) == 0) {
                ring_ok_ = true;
                backend_ = PM_IO_URING;
            }
        }
#else
        (void)backend;
#endif
    }

    void shutdown() {
#ifdef PM_HAS_LIBURING
        if (ring_ok_) { io_uring_queue_exit(&ring_); ring_ok_ = false; }
#endif
    }

    pm_io_backend_t active_backend() const noexcept { return backend_; }

    struct BatchStats {
        uint64_t nvme_bytes = 0;
        uint64_t cache_hits = 0;
        uint64_t cache_misses = 0;
    };

    // Fetch `count` entities from `store`, transparently hitting `cache` (if non-null & enabled)
    // at 4 KiB page granularity and issuing batched O_DIRECT reads ONLY for cache-missed pages!
    void fetch_entities(const SharedDiskStore& store,
                        UnifiedS3FifoCache* cache,
                        const uint32_t* doc_ids,
                        int count,
                        std::vector<const uint8_t*>& out_ptrs,
                        BatchStats& st) {
        out_ptrs.resize(static_cast<size_t>(std::max(0, count)));
        if (count <= 0) return;

        if (store.in_ram) {
            for (int i = 0; i < count; ++i) {
                out_ptrs[i] = store.ram_map.ptr + store.meta[doc_ids[i]].offset;
            }
            return;
        }

        size_t total_bytes = 0;
        for (int i = 0; i < count; ++i) total_bytes += store.meta[doc_ids[i]].aligned_bytes;
        if (total_bytes > bounce_.size()) bounce_.resize(total_bytes, kPageSize);

        struct MissPage {
            uint64_t file_off;
            uint64_t page_idx;
            size_t   bounce_off;
        };
        struct MissSpan {
            uint64_t file_off;
            uint32_t bytes;
            size_t   bounce_off;
        };

        std::vector<MissPage> missed_pages;
        std::vector<MissSpan> missed_spans;
        const bool use_cache = (cache != nullptr && cache->enabled());

        size_t off = 0;
        for (int i = 0; i < count; ++i) {
            const auto& e = store.meta[doc_ids[i]];
            out_ptrs[i] = bounce_.data() + off;
            if (!use_cache) {
                missed_spans.push_back({e.offset, e.aligned_bytes, off});
            } else {
                uint32_t n_pages = e.aligned_bytes / kPageSize;
                uint64_t base_pidx = e.offset / kPageSize;
                uint32_t span_start = 0;
                bool in_span = false;
                for (uint32_t p = 0; p < n_pages; ++p) {
                    size_t p_boff = off + static_cast<size_t>(p) * kPageSize;
                    if (cache->get_page(store.file_id, base_pidx + p, store.tier_tag, bounce_.data() + p_boff)) {
                        st.cache_hits++;
                        if (in_span) {
                            missed_spans.push_back({
                                e.offset + static_cast<uint64_t>(span_start) * kPageSize,
                                (p - span_start) * static_cast<uint32_t>(kPageSize),
                                off + static_cast<size_t>(span_start) * kPageSize
                            });
                            in_span = false;
                        }
                    } else {
                        st.cache_misses++;
                        missed_pages.push_back({e.offset + static_cast<uint64_t>(p) * kPageSize, base_pidx + p, p_boff});
                        if (!in_span) { span_start = p; in_span = true; }
                    }
                }
                if (in_span) {
                    missed_spans.push_back({
                        e.offset + static_cast<uint64_t>(span_start) * kPageSize,
                        (n_pages - span_start) * static_cast<uint32_t>(kPageSize),
                        off + static_cast<size_t>(span_start) * kPageSize
                    });
                }
            }
            off += e.aligned_bytes;
        }

        if (missed_spans.empty()) return;

#ifdef PM_HAS_LIBURING
        if (ring_ok_ && backend_ == PM_IO_URING) {
            int n_sp = static_cast<int>(missed_spans.size());
            int done = 0, sub = 0;
            while (done < n_sp) {
                int batch = 0;
                while (sub < n_sp && (sub - done) < static_cast<int>(qd_)) {
                    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
                    if (!sqe) break;
                    const auto& sp = missed_spans[sub];
                    io_uring_prep_read(sqe, store.fd, bounce_.data() + sp.bounce_off, sp.bytes, sp.file_off);
                    io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(sub));
                    st.nvme_bytes += sp.bytes;
                    sub++; batch++;
                }
                if (batch > 0) io_uring_submit(&ring_);
                struct io_uring_cqe* cqe = nullptr;
                int ret = io_uring_wait_cqe(&ring_, &cqe);
                if (ret < 0) { std::perror("io_uring_wait_cqe"); std::abort(); }
                unsigned head; int seen = 0;
                io_uring_for_each_cqe(&ring_, head, cqe) {
                    if (cqe->res < 0) { std::cerr << "io_uring read err: " << strerror(-cqe->res) << "\n"; std::abort(); }
                    seen++;
                }
                io_uring_cq_advance(&ring_, seen);
                done += seen;
            }
        } else
#endif
        {
            for (const auto& sp : missed_spans) {
                ssize_t r = ::pread(store.fd, bounce_.data() + sp.bounce_off, sp.bytes, static_cast<off_t>(sp.file_off));
                if (r < 0) { std::perror("pread"); std::abort(); }
                st.nvme_bytes += sp.bytes;
            }
        }

        if (use_cache) {
            for (const auto& mp : missed_pages) {
                cache->put_page(store.file_id, mp.page_idx, store.tier_tag, bounce_.data() + mp.bounce_off);
            }
        }
    }

private:
    pm_io_backend_t    backend_ = PM_IO_PREAD_POOL;
    uint32_t           qd_ = 64;
    AlignedBuf<uint8_t> bounce_;
#ifdef PM_HAS_LIBURING
    struct io_uring    ring_{};
    bool               ring_ok_ = false;
#endif
};

} // namespace pm_certvec
