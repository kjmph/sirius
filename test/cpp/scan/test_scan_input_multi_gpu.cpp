/*
 * Copyright 2026, Sirius Contributors.
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

#include "operator/operator_test_utils.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <catch.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <pipeline/gpu_pipeline_task.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace {

void require_two_gpus()
{
  int count = 0;
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
  if (count < 2) { SKIP("Cached scan placement requires two GPUs"); }
}

struct scan_input_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager =
    sirius::test::operator_utils::initialize_memory_manager(2);

  ~scan_input_fixture()
  {
    manager->shutdown();
    sirius::converter_registry::shutdown();
  }

  cucascade::memory::memory_space* gpu(int id)
  {
    return manager->get_memory_space(cucascade::memory::Tier::GPU, id);
  }

  std::shared_ptr<cucascade::data_batch> make_pin(int device, const std::vector<int64_t>& values)
  {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    auto* space = gpu(device);
    auto stream = space->acquire_stream();
    std::shared_ptr<cudf::column> column =
      cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                static_cast<cudf::size_type>(values.size()),
                                cudf::mask_state::UNALLOCATED,
                                stream,
                                space->get_default_allocator());
    REQUIRE(cudaMemcpyAsync(column->mutable_view().data<int64_t>(),
                            values.data(),
                            values.size() * sizeof(int64_t),
                            cudaMemcpyHostToDevice,
                            stream.get()) == cudaSuccess);
    // Raw pins are view-backed. Record the writer on its non-default stream, without
    // synchronizing it here: remote preparation must honor the writer event.
    auto rep = std::make_unique<cucascade::gpu_table_representation>(
      cudf::table_view{{column->view()}}, column, column->alloc_size(), *space, stream);
    return cucascade::data_batch::make(sirius::get_next_batch_id(), std::move(rep));
  }
};

std::vector<int64_t> read_values(const cucascade::data_batch& batch, ::cuda::stream_ref stream)
{
  auto ro     = batch.to_read_only();
  auto column = sirius::get_cudf_table_view(ro).column(0);
  cudaPointerAttributes attributes{};
  REQUIRE(cudaPointerGetAttributes(&attributes, column.data<int64_t>()) == cudaSuccess);
  REQUIRE(attributes.device == ro.get_memory_space()->get_device_id());
  REQUIRE(cudaStreamWaitEvent(stream.get(), ro.get_writer_event(), 0) == cudaSuccess);
  std::vector<int64_t> values(column.size());
  REQUIRE(cudaMemcpyAsync(values.data(),
                          column.data<int64_t>(),
                          values.size() * sizeof(int64_t),
                          cudaMemcpyDeviceToHost,
                          stream.get()) == cudaSuccess);
  stream.sync();
  return values;
}

}  // namespace

TEST_CASE("cached GPU scan prepares input on the requested device", "[scan_prepare][multi_gpu]")
{
  require_two_gpus();
  const auto source_device = GENERATE(0, 1);
  const auto target_device = GENERATE(0, 1);
  const auto has_filter    = GENERATE(false, true);
  CAPTURE(source_device, target_device, has_filter);
  const std::vector<int64_t> values{3, -5, 17, 42};
  scan_input_fixture fixture;
  auto source = fixture.make_pin(source_device, values);
  const void* original_data;
  {
    auto ro       = source->to_read_only();
    original_data = sirius::get_cudf_table_view(ro).column(0).head();
    REQUIRE(ro.get_writer_event() != nullptr);
  }

  auto input                = std::make_unique<sirius::op::scan::scan_operator_input>(source);
  auto* split               = input.get();
  split->row_filter_pending = has_filter;
  sirius::pipeline::gpu_pipeline_task_local_state state(std::move(input));
  auto* target = fixture.gpu(target_device);
  CHECK(state.get_estimated_bytes_to_materialize_input(target) ==
        (source_device == target_device ? 0 : values.size() * sizeof(int64_t)));

  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{target_device}};
  auto stream = target->acquire_stream();
  split->prepare_for_processing(target, stream);
  REQUIRE(split->gpu_memory_space == target);
  auto prepared = split->get_cached_batch();
  REQUIRE(prepared->to_read_only().get_memory_space() == target);
  REQUIRE((prepared == source) == (source_device == target_device));
  REQUIRE(read_values(*prepared, stream) == values);
  CHECK(state.get_estimated_bytes_to_materialize_input(target) == 0);

  // Re-entry after an operator OOM must reuse the already prepared batch.
  split->prepare_for_processing(target, stream);
  REQUIRE(split->get_cached_batch() == prepared);
  split->prepare_for_processing(nullptr, stream);
  REQUIRE(split->get_cached_batch() == prepared);

  // A subsequent scan on the home GPU must still see the original pin, unchanged.
  rmm::cuda_set_device_raii source_guard{rmm::cuda_device_id{source_device}};
  auto source_stream = fixture.gpu(source_device)->acquire_stream();
  sirius::op::scan::scan_operator_input local(source);
  local.prepare_for_processing(fixture.gpu(source_device), source_stream);
  REQUIRE(local.get_cached_batch() == source);
  REQUIRE(sirius::get_cudf_table_view(source->to_read_only()).column(0).head() == original_data);
  REQUIRE(read_values(*source, source_stream) == values);
}

TEST_CASE("cached GPU scan retains its source when cloning fails", "[scan_prepare][multi_gpu]")
{
  require_two_gpus();
  const std::vector<int64_t> values{3, -5, 17, 42};
  scan_input_fixture fixture;
  auto source = fixture.make_pin(1, values);
  sirius::op::scan::scan_operator_input input(source);
  auto& registry = sirius::converter_registry::get();
  using gpu_rep  = cucascade::gpu_table_representation;
  REQUIRE((registry.unregister_converter<gpu_rep, gpu_rep>()));
  registry.register_converter<gpu_rep, gpu_rep>(
    [&](cucascade::idata_representation&,
        const cucascade::memory::memory_space*,
        ::cuda::stream_ref,
        cucascade::memory::reservation*) -> std::unique_ptr<cucascade::idata_representation> {
      CHECK(source->get_state() == cucascade::batch_state::read_only);
      throw rmm::out_of_memory("Injected cached scan clone failure");
    });

  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{0}};
  auto stream = fixture.gpu(0)->acquire_stream();
  REQUIRE_THROWS_AS(input.prepare_for_processing(fixture.gpu(0), stream), rmm::out_of_memory);
  REQUIRE(input.get_cached_batch() == source);
  REQUIRE(source->get_state() == cucascade::batch_state::idle);
  REQUIRE(source->to_read_only().get_memory_space() == fixture.gpu(1));

  sirius::converter_registry::reset_for_testing();
  sirius::converter_registry::initialize();
  input.prepare_for_processing(fixture.gpu(0), stream);
  REQUIRE(input.gpu_memory_space == fixture.gpu(0));
  REQUIRE(input.get_cached_batch() != source);
  REQUIRE(read_values(*input.get_cached_batch(), stream) == values);
}
