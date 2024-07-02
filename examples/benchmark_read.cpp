#include <fast_matrix_market/fast_matrix_market.hpp>
#include <time.h>
#include <fstream>
#include <fmt/ranges.h>
#include <cstdlib>

#include <chrono>

double gettime() {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return ((double) time.tv_sec) + ((double) 1e-9) * time.tv_nsec;
}

template <typename I, typename V>
struct triplet_matrix {
    int64_t nrows = 0, ncols = 0;
    std::vector<I> rows, cols;
    std::vector<V> vals;       // or int64_t, float, std::complex<double>, etc.
};

template <typename T, typename I>
void benchmark_fastmm_read(const std::string& file_name, int num_threads) {
  bool multi_threaded = false;
  if (num_threads > 1) {
    multi_threaded = true;
  }

  fast_matrix_market::read_options options{};
  options.generalize_symmetry = false;
  options.parallel_ok = multi_threaded;
  options.num_threads = num_threads;

  fast_matrix_market::matrix_market_header header;
  triplet_matrix<I, T> triplet;

  auto begin = std::chrono::high_resolution_clock::now();

  std::ifstream stream(file_name);

  fast_matrix_market::read_matrix_market_triplet(stream, header, triplet.rows, triplet.cols, triplet.vals, options);
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  double nbytes = (triplet.rows.size() + triplet.cols.size())*sizeof(I) + triplet.vals.size() * sizeof(T);
  double gbytes = nbytes / 1024 / 1024 / 1024;

  double gbytes_s = gbytes / duration;

  fmt::print("FORPARSER: {},{},{}\n", file_name, duration, gbytes_s);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: ./benchmark_read [file_name.mtx] [file_name.h5] [optional: numthreads (default 1)]\n");
    return 1;
  }

  char* file_name = argv[1];

  char* binsparse_file = argv[2];

  int num_threads = 1;
  if (argc >= 4) {
    num_threads = std::atoi(argv[3]);
  }

  benchmark_fastmm_read<float, int64_t>(file_name, num_threads);

  return 0;
}
