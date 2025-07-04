# Copyright 2025 Google LLC
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

# pylint: disable=protected-access,bad-continuation

import importlib
import pytest

from google.cloud import aiplatform
import vertexai
from google.cloud.aiplatform import initializer as aiplatform_initializer


_TEST_PROJECT = "test-project"
_TEST_LOCATION = "us-central1"


pytestmark = pytest.mark.usefixtures("google_auth_mock")


class TestGenAiClient:
    """Unit tests for the GenAI client."""

    def setup_method(self):
        importlib.reload(aiplatform_initializer)
        importlib.reload(aiplatform)
        importlib.reload(vertexai)
        vertexai.init(
            project=_TEST_PROJECT,
            location=_TEST_LOCATION,
        )

    @pytest.mark.usefixtures("google_auth_mock")
    def test_genai_client(self):
        test_client = vertexai.Client(project=_TEST_PROJECT, location=_TEST_LOCATION)
        assert test_client is not None
        assert test_client._api_client.vertexai
        assert test_client._api_client.project == _TEST_PROJECT
        assert test_client._api_client.location == _TEST_LOCATION

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("google_auth_mock")
    async def test_async_client(self):
        test_client = vertexai.Client(project=_TEST_PROJECT, location=_TEST_LOCATION)
        assert isinstance(test_client.aio, vertexai._genai.client.AsyncClient)

    @pytest.mark.usefixtures("google_auth_mock")
    def test_types(self):
        assert vertexai.types is not None
        assert vertexai.types.LLMMetric is not None
