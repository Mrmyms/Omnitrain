#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <cstring>

namespace py = pybind11;

struct BusHeader {
    pthread_mutex_t mutex;
    int write_ptr;
    int max_tokens;
    int token_dim;
    int modal_id_len;
};

class NativeTokenBus {
public:
    NativeTokenBus(int max_tokens, int token_dim, int modal_id_len, std::string sid, bool create)
        : max_tokens(max_tokens), token_dim(token_dim), modal_id_len(modal_id_len), sid(sid) {
        
        size_t header_size = sizeof(BusHeader);
        size_t data_size = max_tokens * token_dim * sizeof(float);
        size_t ts_size = max_tokens * sizeof(double);
        size_t id_size = max_tokens * modal_id_len;
        size_t total_size = header_size + data_size + ts_size + id_size;

        std::string shm_name = "/omni_" + sid;
        int fd;

        if (create) {
            shm_unlink(shm_name.c_str());
            fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
            if (fd == -1) throw std::runtime_error("shm_open create failed");
            if (ftruncate(fd, total_size) == -1) throw std::runtime_error("ftruncate failed");
        } else {
            fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
            if (fd == -1) throw std::runtime_error("shm_open attach failed");
        }

        base_ptr = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (base_ptr == MAP_FAILED) throw std::runtime_error("mmap failed");
        close(fd);

        header = (BusHeader*)base_ptr;
        data_ptr = (float*)((char*)base_ptr + header_size);
        ts_ptr = (double*)((char*)data_ptr + data_size);
        id_ptr = (char*)((char*)ts_ptr + ts_size);

        if (create) {
            pthread_mutexattr_t attr;
            pthread_mutexattr_init(&attr);
            pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
            pthread_mutex_init(&header->mutex, &attr);
            header->write_ptr = 0;
            header->max_tokens = max_tokens;
            header->token_dim = token_dim;
            header->modal_id_len = modal_id_len;
        }
    }

    void publish(py::array_t<float> data, double timestamp, std::string modal_id) {
        pthread_mutex_lock(&header->mutex);
        
        int idx = header->write_ptr;
        auto r = data.unchecked<2>(); // Assume (N, token_dim) or (1, token_dim)
        int rows = r.shape(0);

        for (int i = 0; i < rows; ++i) {
            int current_idx = (idx + i) % header->max_tokens;
            
            // Raw copy of token data
            float* dst_data = data_ptr + (current_idx * header->token_dim);
            std::memcpy(dst_data, data.data(i, 0), header->token_dim * sizeof(float));
            
            ts_ptr[current_idx] = timestamp;

            // Copy Modal ID
            char* dst_id = id_ptr + (current_idx * header->modal_id_len);
            std::memset(dst_id, 0, header->modal_id_len);
            std::strncpy(dst_id, modal_id.c_str(), header->modal_id_len - 1);
        }

        header->write_ptr = (idx + rows) % header->max_tokens;
        pthread_mutex_unlock(&header->mutex);
    }

    py::list get_window(double start_time, double end_time) {
        py::list results;
        pthread_mutex_lock(&header->mutex);

        for (int i = 0; i < header->max_tokens; ++i) {
            if (ts_ptr[i] >= start_time && ts_ptr[i] <= end_time) {
                py::dict entry;
                
                // Copy data to numpy
                auto data_array = py::array_t<float>(header->token_dim);
                std::memcpy(data_array.mutable_data(), data_ptr + (i * header->token_dim), header->token_dim * sizeof(float));
                
                entry["data"] = data_array;
                entry["timestamp"] = ts_ptr[i];
                entry["modal_id"] = std::string(id_ptr + (i * header->modal_id_len));
                results.append(entry);
            }
        }

        pthread_mutex_unlock(&header->mutex);
        return results;
    }

    void cleanup(bool unlink) {
        if (base_ptr) {
            // size_t total_size = ... calculate again if needed
            // munmap(base_ptr, total_size);
            base_ptr = nullptr;
        }
        if (unlink) {
            shm_unlink(("/omni_" + sid).c_str());
        }
    }

private:
    int max_tokens, token_dim, modal_id_len;
    std::string sid;
    void* base_ptr = nullptr;
    BusHeader* header = nullptr;
    float* data_ptr = nullptr;
    double* ts_ptr = nullptr;
    char* id_ptr = nullptr;
};

PYBIND11_MODULE(omni_bus_core, m) {
    py::class_<NativeTokenBus>(m, "NativeTokenBus")
        .def(py::init<int, int, int, std::string, bool>())
        .def("publish", &NativeTokenBus::publish)
        .def("get_window", &NativeTokenBus::get_window)
        .def("cleanup", &NativeTokenBus::cleanup);
}
