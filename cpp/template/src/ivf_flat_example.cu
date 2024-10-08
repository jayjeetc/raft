/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <raft/neighbors/ivf_flat.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

#include <cstdint>
#include <optional>

void ivf_flat_build_search_simple(raft::device_resources const& dev_resources,
                                  raft::device_matrix_view<const float, int64_t> dataset,
                                  raft::device_matrix_view<const float, int64_t> queries,
                                  int64_t topk)
{
  using namespace raft::neighbors;

  ivf_flat::index_params index_params;
  index_params.n_lists                  = 1024;
  index_params.kmeans_trainset_fraction = 0.1;
  index_params.metric                   = raft::distance::DistanceType::L2Expanded;

  auto s = std::chrono::high_resolution_clock::now();
  auto index = ivf_flat::build(dev_resources, index_params, dataset);
  auto e = std::chrono::high_resolution_clock::now();
  std::cout << "IVF-Flat index built in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() << " ms" << std::endl;
  std::cout << "Number of clusters " << index.n_lists() << ", number of vectors added to index "
            << index.size() << std::endl;
  
  // Create output arrays.
  int64_t n_queries = queries.extent(0);
  auto neighbors    = raft::make_device_matrix<int64_t>(dev_resources, n_queries, topk);
  auto distances    = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // Set search parameters.
  ivf_flat::search_params search_params;
  search_params.n_probes = 50;

  // Search K nearest neighbors for each of the queries.
  s = std::chrono::high_resolution_clock::now();
  ivf_flat::search(
    dev_resources, search_params, index, queries, neighbors.view(), distances.view());
  e = std::chrono::high_resolution_clock::now();
  std::cout << "IVF-Flat search completed in "
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
  rmm::mr::managed_memory_resource managed_mr;
  rmm::mr::set_current_device_resource(&managed_mr);

  // Create input arrays.
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
  std::cout << "Data generated in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() << " ms" << std::endl;

  // Simple build and search example.
  ivf_flat_build_search_simple(dev_resources,
                               raft::make_const_mdspan(dataset.view()),
                               raft::make_const_mdspan(queries.view()),
                               top_k);
}
