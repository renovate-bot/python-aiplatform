# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

def make_parent(parent: str) -> str:
    # Sample function parameter parent in create_batch_prediction_job_sample
    parent = parent

    return parent

def make_batch_prediction_job(display_name: str, model_name: str, instances_format: str, gcs_source_uri: str, predictions_format: str, gcs_destination_output_uri_prefix: str) -> google.cloud.aiplatform_v1alpha1.types.batch_prediction_job.BatchPredictionJob:
    model_parameters_dict = {}
    model_parameters = to_protobuf_value(model_parameters_dict)

    batch_prediction_job = {
        'display_name': display_name,
        # Format: 'projects/{project}/locations/{location}/models/{model_id}'
        'model': model_name,
        'model_parameters': model_parameters,
        'input_config': {
            'instances_format': instances_format,
            'gcs_source': {
                'uris': [gcs_source_uri]
            },
        },
        'output_config': {
            'predictions_format': predictions_format,
            'gcs_destination': {
                'output_uri_prefix': gcs_destination_output_uri_prefix
            },
        },
        'dedicated_resources': {
            'machine_spec': {
                'machine_type': 'n1-standard-2',
                'accelerator_type': aiplatform.gapic.AcceleratorType.NVIDIA_TESLA_K80,
                'accelerator_count': 1
            },
            'starting_replica_count': 1,
            'max_replica_count': 1
        }
    }

    return batch_prediction_job

