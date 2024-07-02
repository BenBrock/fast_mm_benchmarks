function(add_example example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} fast_matrix_market::fast_matrix_market fmt)
endfunction()

add_example(benchmark_read)

