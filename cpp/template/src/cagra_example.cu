/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/neighbors/cagra.cuh>
#include <raft/random/make_blobs.cuh>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cstdint>

void cagra_build_search_simple(raft::device_resources const& dev_resources,
                               raft::device_matrix_view<const float, int64_t> dataset,
                               raft::device_matrix_view<const float, int64_t> queries,
                               int64_t topk)
{
  using namespace raft::neighbors;
  int64_t n_queries = queries.extent(0);

  // create output arrays
  auto neighbors = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // use default index parameters
  cagra::index_params index_params;

  auto s = std::chrono::high_resolution_clock::now();
  auto index = cagra::build<float, uint32_t>(dev_resources, index_params, dataset);
  auto e = std::chrono::high_resolution_clock::now();
  std::cout << "CAGRA index built in " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
            << " ms" << std::endl;

  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << index.graph_degree() << ", graph size ["
            << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;

  cagra::search_params search_params;
  s = std::chrono::high_resolution_clock::now();
  cagra::search<float, uint32_t>(
    dev_resources, search_params, index, queries, neighbors.view(), distances.view());
  e = std::chrono::high_resolution_clock::now();
  std::cout << "CAGRA search completed in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() << " ms" << std::endl;

  print_results(dev_resources, neighbors.view(), distances.view());
}

int main(int argc, char** argv)
{
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <n_samples> <n_queries> <n_dim> <top_k>" << std::endl;
    return 1;
  }
  
  // Set the device memory resource to the CUDA memory resource.
  raft::device_resources dev_resources;
  rmm::mr::cuda_memory_resource cuda_mr;
  rmm::mr::set_current_device_resource(&cuda_mr);

  int64_t n_samples = std::atoi(argv[1]);
  int64_t n_queries = std::atoi(argv[2]);
  int64_t n_dim     = std::atoi(argv[3]);
  int64_t top_k     = std::atoi(argv[4]);
  std::cout << "n_samples: " << n_samples << ", n_queries: " << n_queries << ", n_dim: " << n_dim
            << ", top_k: " << top_k << std::endl;

  // Generate the dataset and queries.
  auto dataset      = raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim);
  auto queries      = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);
  auto s = std::chrono::high_resolution_clock::now();
  generate_dataset(dev_resources, dataset.view(), queries.view());
  auto e = std::chrono::high_resolution_clock::now();
  std::cout << "Data generated in " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
            << " ms" << std::endl;

  // Simple build and search example.
  cagra_build_search_simple(dev_resources,
                            raft::make_const_mdspan(dataset.view()),
                            raft::make_const_mdspan(queries.view()),
                            top_k);
}
