# Copyright 2024 Google LLC
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
"""Classes for working with reasoning engines."""

# We just want to re-export certain classes
# pylint: disable=g-multiple-import,g-importing-member
from vertexai.reasoning_engines._reasoning_engines import (
    Queryable,
    ReasoningEngine,
)
from vertexai.preview.reasoning_engines.templates.ag2 import (
    AG2Agent,
)
from vertexai.preview.reasoning_engines.templates.langchain import (
    LangchainAgent,
)
from vertexai.preview.reasoning_engines.templates.langgraph import (
    LanggraphAgent,
)

__all__ = (
    "AG2Agent",
    "LangchainAgent",
    "LanggraphAgent",
    "Queryable",
    "ReasoningEngine",
)
