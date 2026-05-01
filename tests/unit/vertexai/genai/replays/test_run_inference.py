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
# pylint: disable=protected-access,bad-continuation,missing-function-docstring

import pytest

from tests.unit.vertexai.genai.replays import pytest_helper
from vertexai._genai import types
from google.genai import types as genai_types

pytest.importorskip(
    "google.adk", reason="google-adk not installed, skipping ADK agent tests"
)
from google.adk.agents import (  # noqa: E402
    LlmAgent,
)  # pylint: disable=g-import-not-at-top,g-bad-import-order


def test_inference_with_eval_cases_multi_turn_agent_data(client):
    """Tests run_inference with multi-turn agent_data in eval_cases.

    Verifies that run_inference() accepts an EvaluationDataset with
    eval_cases containing agent_data (no eval_dataset_df). The agent_data
    has 2 turns: turn 0 is a completed user+agent exchange (history),
    turn 1 is a new user query. The agent should see the history and
    respond to the final query in context.
    """
    agent = LlmAgent(
        name="test_agent",
        model="gemini-2.5-flash",
        instruction="You are a helpful assistant. Answer questions concisely.",
    )

    eval_case = types.EvalCase(
        agent_data=types.evals.AgentData(
            turns=[
                types.evals.ConversationTurn(
                    turn_index=0,
                    events=[
                        types.evals.AgentEvent(
                            author="user",
                            content=genai_types.Content(
                                role="user",
                                parts=[genai_types.Part(text="My name is Alice.")],
                            ),
                        ),
                        types.evals.AgentEvent(
                            author="test_agent",
                            content=genai_types.Content(
                                role="model",
                                parts=[
                                    genai_types.Part(
                                        text="Hello Alice! How can I help you?"
                                    )
                                ],
                            ),
                        ),
                    ],
                ),
                types.evals.ConversationTurn(
                    turn_index=1,
                    events=[
                        types.evals.AgentEvent(
                            author="user",
                            content=genai_types.Content(
                                role="user",
                                parts=[genai_types.Part(text="What is my name?")],
                            ),
                        ),
                    ],
                ),
            ],
        ),
    )
    eval_dataset = types.EvaluationDataset(eval_cases=[eval_case])

    inference_result = client.evals.run_inference(
        agent=agent,
        src=eval_dataset,
    )
    assert isinstance(inference_result, types.EvaluationDataset)
    assert inference_result.eval_dataset_df is not None
    assert "agent_data" in inference_result.eval_dataset_df.columns


pytestmark = pytest_helper.setup(
    file=__file__,
    globals_for_file=globals(),
    test_method="evals.run_inference",
)
