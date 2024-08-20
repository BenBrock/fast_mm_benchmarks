#include <fast_matrix_market/fast_matrix_market.hpp>
#include <time.h>
#include <fstream>
#include <fmt/ranges.h>
#include <cstdlib>
#include <binsparse/binsparse.h>

#include <concepts>
#include <chrono>
#include <span>
#include <ranges>

#include <unistd.h>

template <std::ranges::range R, std::ranges::range U>
bool ranges_equal(R&& r, U&& u) {
  if (std::ranges::size(r) != std::ranges::size(u)) {
    return false;
  }
  auto rit = std::ranges::begin(r);
  auto uit = std::ranges::begin(u);

  for ( ; rit != std::ranges::end(r) ;) {
    if (*rit != *uit) {
      return false;
    }
    ++rit;
    ++uit;
  }
  return true;
}

fast_matrix_market::symmetry_type get_symmetry_type(bsp_structure_t structure) {
  if (structure == BSP_GENERAL) {
    return fast_matrix_market::symmetry_type::general;
  } else if (structure == BSP_SYMMETRIC) {
    return fast_matrix_market::symmetry_type::symmetric;
  } else if (structure == BSP_SKEW_SYMMETRIC) {
    return fast_matrix_market::symmetry_type::skew_symmetric;
  } else if (structure == BSP_HERMITIAN) {
    return fast_matrix_market::symmetry_type::hermitian;
  } else {
    assert(false);
  }
}

fast_matrix_market::field_type get_field_type(bsp_type_t type) {
  if (type >= BSP_UINT8 && type <= BSP_INT64) {
    return fast_matrix_market::field_type::integer;
  } else if (type >= BSP_FLOAT32 && type <= BSP_FLOAT64) {
    return fast_matrix_market::field_type::real;
  } else if (type >= BSP_COMPLEX_FLOAT32 && type <= BSP_COMPLEX_FLOAT64) {
    return fast_matrix_market::field_type::complex;
  } else {
    assert(false);
  }
}

template <typename T>
class cspan {
public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator = T*;
  using const_iterator = const T*;

  cspan(iterator first, size_type count) : data_(first), size_(count) {}

  iterator begin() const noexcept {
    return data_;
  }

  iterator end() const noexcept {
    return data_ + size_;
  }

  const_iterator cbegin() const noexcept {
    return begin();
  }

  const_iterator cend() const noexcept {
    return end();
  }

  size_type size() const noexcept {
    return size_;
  }

private:
  T* data_;
  std::size_t size_;
};

void flush_cache() {
#ifdef __APPLE__
  system("bash -c \"sync && sudo purge\"");
#elif __linux__
  system("bash -c \"sync\" && sudo sh -c \"/usr/bin/echo 3 > "
         "/proc/sys/vm/drop_caches\"");
#else
  static_assert(false);
#endif
  usleep(100000);
}

void flush_writes() {
#ifdef __APPLE__
  system("bash -c \"sync\"");
#elif __linux__
  system("bash -c \"sync\"");
#else
  static_assert(false);
#endif
}

template <typename T>
struct convert_to_cpp_fp {
  using type = T;
};

template <>
struct convert_to_cpp_fp<_Complex float> {
  using type = std::complex<float>;
};

template <>
struct convert_to_cpp_fp<_Complex double> {
  using type = std::complex<double>;
};

template <typename T>
using convert_to_cpp_fp_t = typename convert_to_cpp_fp<T>::type;

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
void verify_fastmm_write(bsp_matrix_t matrix, const std::string& file_name, int num_threads, bool flush_write = true, bool print = true) {
  bool multi_threaded = false;
  if (num_threads > 1) {
    multi_threaded = true;
  }

  assert(matrix.format == BSP_COO);

  fast_matrix_market::write_options options{};
  options.parallel_ok = multi_threaded;
  options.num_threads = num_threads;

  fast_matrix_market::matrix_market_header header;

  header.nrows = matrix.nrows;
  header.ncols = matrix.ncols;
  header.nnz = matrix.nnz;

  header.symmetry = get_symmetry_type(matrix.structure);

  if (matrix.is_iso) {
    header.field = fast_matrix_market::field_type::pattern;
  } else {
    header.field = get_field_type(matrix.values.type);
  }

  cspan<const T> vals((T*) matrix.values.data, matrix.values.size);
  cspan<const I> rows((I*) matrix.indices_0.data, matrix.indices_0.size);
  cspan<const I> cols((I*) matrix.indices_1.data, matrix.indices_1.size);

  assert((matrix.indices_0.size == matrix.indices_1.size) && (matrix.indices_1.size == matrix.nnz));

  if (!matrix.is_iso) {
    assert(matrix.indices_0.size == matrix.values.size);
  }

  auto begin = std::chrono::high_resolution_clock::now();

  std::ofstream stream(file_name);

  fast_matrix_market::write_matrix_market_triplet(stream, header, rows, cols, vals, options);

  stream.close();

  std::ifstream istream(file_name);
  fast_matrix_market::matrix_market_header iheader;

  fast_matrix_market::read_options ioptions{};
  ioptions.generalize_symmetry = false;

  triplet_matrix<I, T> triplet;

  fast_matrix_market::read_matrix_market_triplet(istream, iheader, triplet.rows, triplet.cols, triplet.vals, ioptions);

  istream.close();

  if (!ranges_equal(triplet.rows, rows) || !ranges_equal(triplet.cols, cols)
      || (header.field != fast_matrix_market::field_type::pattern && !ranges_equal(triplet.vals, vals))) {
    printf("FAIL! Arrays not equal.\n");
    fflush(stdout);
    assert(false);
  } else {
    printf("OK!\n");
    fflush(stdout);
  }

  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  double nbytes = (rows.size() + cols.size())*sizeof(I) + vals.size() * sizeof(T);
  double gbytes = nbytes / 1024 / 1024 / 1024;

  double gbytes_s = gbytes / duration;

  fmt::print("FORPARSER: {},{},{}\n", file_name, duration, gbytes_s);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: ./write_verify [output_file_name.mtx] [input_file_name.h5] [optional: numthreads (default 1)]\n");
    return 1;
  }

  char* file_name = argv[1];

  char* binsparse_file = argv[2];

  bsp_matrix_t mat = bsp_read_matrix(binsparse_file, NULL);

  assert(mat.values.size != 0);
  assert(mat.indices_0.size != 0);

  auto value_ptr = binsparse::__detail::get_typed_ptr(mat.values);
  auto index_ptr = binsparse::__detail::get_typed_ptr(mat.indices_0);

  int num_threads = 1;
  if (argc >= 4) {
    num_threads = std::atoi(argv[3]);
  }

  bool flush_writes = true;

  std::visit([&](auto* v, auto* i) {
    using T = std::remove_pointer_t<decltype(v)>;
    using I = std::remove_pointer_t<decltype(i)>;

    using T_ = convert_to_cpp_fp_t<T>;
    if constexpr(std::integral<I>) {
      verify_fastmm_write<T_, I>(mat, file_name, num_threads, flush_writes);
    }
  }, value_ptr, index_ptr);

  bsp_destroy_matrix_t(mat);

  return 0;
}
