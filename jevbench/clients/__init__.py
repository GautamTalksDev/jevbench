"""Client layer — shared Decision interface + four backends."""

from jevbench.clients.adapter import AdapterClient, AdapterClientConfig
from jevbench.clients.base import (
    SENTINEL_DOC_KEY,
    ChoiceQuestion,
    Decision,
    DecisionClient,
    Kind,
    NoulQuestion,
    Question,
    ScoreQuestion,
    SystemOneRequest,
)
from jevbench.clients.jev import JevClient, JevClientConfig
from jevbench.clients.prefill import (
    PrefillClient,
    PrefillClientConfig,
    PrefillCompletion,
    build_sentinel_mapping,
)
from jevbench.clients.gliclass import GLiClassClient, GLiClassClientConfig
from jevbench.clients.trivial import (
    KeywordRule,
    TrivialClient,
    TrivialClientConfig,
    TrivialQuestionSpec,
)

__all__ = [
    "SENTINEL_DOC_KEY",
    "AdapterClient",
    "AdapterClientConfig",
    "ChoiceQuestion",
    "Decision",
    "DecisionClient",
    "GLiClassClient",
    "GLiClassClientConfig",
    "JevClient",
    "JevClientConfig",
    "KeywordRule",
    "Kind",
    "NoulQuestion",
    "PrefillClient",
    "PrefillClientConfig",
    "PrefillCompletion",
    "Question",
    "ScoreQuestion",
    "SystemOneRequest",
    "TrivialClient",
    "TrivialClientConfig",
    "TrivialQuestionSpec",
    "build_sentinel_mapping",
]
