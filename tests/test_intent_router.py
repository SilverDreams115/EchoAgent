"""
Comprehensive tests for echo.ui.intent_router.

Coverage:
- All merge forms: bare verb, "todo" forms, with optional destination suffix
- All cherry-pick forms: artefact keywords, "desde" preposition, incorpora variant
- Extract artefacts: Spanish/English synonyms
- Edge cases: false-positive guards, ambiguous sentences → conversation
- Ordering guarantee: "todo" always wins over artefact-bearing patterns
- Contextual source resolution: "esta rama", "la actual", "current branch", etc.
"""

from __future__ import annotations

import pytest
from echo.ui.intent_router import (
    extract_artefacts,
    is_contextual_source_ref,
    resolve_contextual_source,
    route,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def intent(text: str) -> str:
    return route(text)[0]


def values(text: str) -> dict:
    return route(text)[1]


# ---------------------------------------------------------------------------
# Merge — bare verb forms (no "todo", no destination)
# ---------------------------------------------------------------------------

class TestMergeBareVerb:
    def test_merge_en(self):
        assert intent("merge experiment") == "branch_merge"

    def test_merge_en_source(self):
        assert values("merge experiment")["source"] == "experiment"

    def test_fusiona(self):
        assert intent("fusiona experiment") == "branch_merge"

    def test_fusionar(self):
        assert intent("fusionar experiment") == "branch_merge"

    def test_combina(self):
        assert intent("combina feature-x") == "branch_merge"

    def test_mezcla(self):
        assert intent("mezcla feature-x") == "branch_merge"

    def test_mezclar(self):
        assert intent("mezclar feature-x") == "branch_merge"

    def test_integra(self):
        assert intent("integra feature-x") == "branch_merge"

    def test_integrar(self):
        assert intent("integrar feature-x") == "branch_merge"

    def test_incorpora(self):
        assert intent("incorpora experiment") == "branch_merge"

    def test_absorbe(self):
        assert intent("absorbe experiment") == "branch_merge"

    def test_haz_merge_de(self):
        assert intent("haz merge de experiment") == "branch_merge"

    def test_haz_un_merge_de(self):
        assert intent("haz un merge de experiment") == "branch_merge"


# ---------------------------------------------------------------------------
# Merge — with optional destination suffix
# ---------------------------------------------------------------------------

class TestMergeWithDestination:
    def test_merge_a_main(self):
        assert intent("merge experiment a main") == "branch_merge"
        assert values("merge experiment a main")["source"] == "experiment"

    def test_mezcla_en_main(self):
        assert intent("mezcla experiment en main") == "branch_merge"
        assert values("mezcla experiment en main")["source"] == "experiment"

    def test_fusiona_con_main(self):
        assert intent("fusiona feature-x con main") == "branch_merge"

    def test_integra_hacia_main(self):
        assert intent("integra feature-x hacia main") == "branch_merge"

    def test_incorpora_a_main(self):
        assert intent("incorpora feature-x a main") == "branch_merge"

    def test_absorbe_en_main(self):
        assert intent("absorbe feature-x en main") == "branch_merge"

    def test_haz_merge_a_main(self):
        assert intent("haz merge de feature-x a main") == "branch_merge"


# ---------------------------------------------------------------------------
# Merge — "todo" forms
# ---------------------------------------------------------------------------

class TestMergeTodo:
    def test_trae_todo_de(self):
        assert intent("trae todo de experiment") == "branch_merge"

    def test_trae_todo_desde(self):
        assert intent("trae todo desde experiment") == "branch_merge"

    def test_trae_todo_lo_de(self):
        assert intent("trae todo lo de experiment") == "branch_merge"

    def test_trae_todo_lo_desde(self):
        assert intent("trae todo lo desde experiment") == "branch_merge"

    def test_trae_todo_util_de(self):
        assert intent("trae todo lo útil de experiment") == "branch_merge"

    def test_trae_todo_importante_de(self):
        assert intent("trae todo lo importante de experiment") == "branch_merge"

    def test_trae_todo_relevante_desde(self):
        assert intent("trae todo lo relevante desde feature-x") == "branch_merge"

    def test_trae_todo_de_a_dest(self):
        assert intent("trae todo de experiment a main") == "branch_merge"
        assert values("trae todo de experiment a main")["source"] == "experiment"

    def test_dame_todo_de(self):
        assert intent("dame todo de experiment") == "branch_merge"

    def test_dame_todo_desde(self):
        assert intent("dame todo desde experiment") == "branch_merge"

    def test_pasame_todo_de(self):
        assert intent("pásame todo de experiment") == "branch_merge"

    def test_pasa_todo_de(self):
        assert intent("pasa todo de experiment a main") == "branch_merge"

    def test_traeme_todo_de(self):
        assert intent("tráeme todo de experiment") == "branch_merge"

    def test_dame_todo_util_de(self):
        assert intent("dame todo lo útil de feature-x") == "branch_merge"

    def test_mezcla_todo_de(self):
        assert intent("mezcla todo de experiment") == "branch_merge"

    def test_mezcla_todo_lo_de(self):
        assert intent("mezcla todo lo de feature-x") == "branch_merge"

    def test_mezcla_todo_lo_util_de(self):
        assert intent("mezcla todo lo útil de feature-x") == "branch_merge"

    def test_mezcla_todo_lo_de_en_main(self):
        assert intent("mezcla todo lo de feature-x en main") == "branch_merge"
        assert values("mezcla todo lo de feature-x en main")["source"] == "feature-x"

    def test_fusiona_todo_de(self):
        assert intent("fusiona todo de experiment") == "branch_merge"

    def test_combina_todo_desde(self):
        assert intent("combina todo desde feature-x") == "branch_merge"

    def test_integra_todo_de(self):
        assert intent("integra todo de feature-x") == "branch_merge"

    def test_incorpora_todo_de(self):
        assert intent("incorpora todo de experiment") == "branch_merge"

    def test_incorpora_todo_lo_util_de(self):
        assert intent("incorpora todo lo útil de experiment") == "branch_merge"

    def test_incorpora_todo_lo_util_de_a_main(self):
        assert intent("incorpora todo lo útil de feature-x a main") == "branch_merge"
        assert values("incorpora todo lo útil de feature-x a main")["source"] == "feature-x"


# ---------------------------------------------------------------------------
# Cherry-pick — artefact-qualified forms
# ---------------------------------------------------------------------------

class TestCherryPickArtefacts:
    def test_trae_decisions_de(self):
        assert intent("trae las decisiones de experiment") == "branch_cherry_pick"

    def test_trae_decisions_desde(self):
        assert intent("trae las decisiones desde experiment") == "branch_cherry_pick"

    def test_trae_findings_de(self):
        assert intent("trae los findings de feature-x") == "branch_cherry_pick"

    def test_trae_findings_desde(self):
        assert intent("trae los findings desde feature-x") == "branch_cherry_pick"

    def test_trae_facts_y_findings_de(self):
        assert intent("trae facts y findings de experiment") == "branch_cherry_pick"

    def test_trae_facts_y_findings_desde(self):
        assert intent("trae facts y findings desde experiment") == "branch_cherry_pick"

    def test_dame_findings_de(self):
        assert intent("dame los findings de feature-x") == "branch_cherry_pick"

    def test_dame_findings_desde(self):
        assert intent("dame los findings desde feature-x") == "branch_cherry_pick"

    def test_dame_errores_de(self):
        assert intent("dame los errores de feature-x") == "branch_cherry_pick"

    def test_quiero_decisions_de(self):
        assert intent("quiero las decisiones de experiment") == "branch_cherry_pick"

    def test_quiero_findings_desde(self):
        assert intent("quiero los findings desde feature-x") == "branch_cherry_pick"

    def test_quiero_solo_decisions_de(self):
        assert intent("quiero solo las decisiones de experiment") == "branch_cherry_pick"

    def test_extrae_decisions_de(self):
        assert intent("extrae las decisiones de experiment") == "branch_cherry_pick"

    def test_extrae_decisions_y_findings_desde(self):
        assert intent("extrae las decisiones y findings desde feature-x") == "branch_cherry_pick"

    def test_importa_findings_de(self):
        assert intent("importa los findings de experiment") == "branch_cherry_pick"

    def test_importa_findings_desde(self):
        assert intent("importa los findings desde experiment") == "branch_cherry_pick"

    def test_incorpora_decisions_de(self):
        assert intent("incorpora las decisiones de experiment") == "branch_cherry_pick"

    def test_incorpora_decisions_desde(self):
        assert intent("incorpora las decisiones desde experiment") == "branch_cherry_pick"

    def test_incorpora_unicamente_decisions_de(self):
        assert intent("incorpora únicamente las decisiones de experiment") == "branch_cherry_pick"

    def test_incorpora_solo_findings_de(self):
        assert intent("incorpora solo los findings de feature-x") == "branch_cherry_pick"

    def test_incorpora_solo_pending_desde(self):
        assert intent("incorpora solo los pending desde feature-x") == "branch_cherry_pick"

    def test_cherry_pick_bare(self):
        assert intent("cherry-pick experiment") == "branch_cherry_pick"

    def test_cherry_pick_with_flags(self):
        assert intent("cherry-pick experiment --decisions --findings") == "branch_cherry_pick"

    def test_cherry_pick_de_bare(self):
        assert intent("cherry-pick de experiment") == "branch_cherry_pick"


# ---------------------------------------------------------------------------
# Cherry-pick — source extracted correctly
# ---------------------------------------------------------------------------

class TestCherryPickSource:
    def test_source_from_trae(self):
        assert values("trae las decisiones de feature-x")["source"] == "feature-x"

    def test_source_from_dame(self):
        assert values("dame los findings desde feature-x")["source"] == "feature-x"

    def test_source_from_incorpora(self):
        assert values("incorpora únicamente las decisiones de feature-x")["source"] == "feature-x"

    def test_source_from_extrae(self):
        assert values("extrae los errores desde experiment")["source"] == "experiment"


# ---------------------------------------------------------------------------
# Cherry-pick — artefacts extracted correctly
# ---------------------------------------------------------------------------

class TestCherryPickArtefactExtraction:
    def test_decisions_extracted(self):
        assert "decisions" in values("trae las decisiones de feature-x")["artefacts"]

    def test_findings_extracted(self):
        assert "findings" in values("dame los findings de feature-x")["artefacts"]

    def test_multiple_artefacts(self):
        a = values("trae facts y findings desde experiment")["artefacts"]
        assert "facts" in a
        assert "findings" in a

    def test_errors_extracted_via_errores(self):
        a = values("dame los errores de feature-x")["artefacts"]
        assert "errors" in a

    def test_pending_extracted_via_pendientes(self):
        a = values("incorpora solo los pendientes desde feature-x")["artefacts"]
        assert "pending" in a

    def test_incorpora_cherry_decisions(self):
        a = values("incorpora únicamente las decisiones de feature-x")["artefacts"]
        assert "decisions" in a


# ---------------------------------------------------------------------------
# extract_artefacts — synonym coverage
# ---------------------------------------------------------------------------

class TestExtractArtefacts:
    def test_decisiones(self):
        assert extract_artefacts("decisiones") == ["decisions"]

    def test_hallazgos(self):
        assert extract_artefacts("hallazgos") == ["findings"]

    def test_resultado(self):
        assert extract_artefacts("resultado") == ["findings"]

    def test_tarea(self):
        assert extract_artefacts("tarea") == ["pending"]

    def test_accion(self):
        assert extract_artefacts("accion") == ["pending"]

    def test_dato(self):
        assert extract_artefacts("dato") == ["facts"]

    def test_datos(self):
        assert extract_artefacts("datos") == ["facts"]

    def test_resumen(self):
        assert extract_artefacts("resumen") == ["summary"]

    def test_fallo(self):
        assert extract_artefacts("fallo") == ["errors"]

    def test_fallos(self):
        assert extract_artefacts("fallos") == ["errors"]

    def test_cambio(self):
        assert extract_artefacts("cambio") == ["changes"]

    def test_modificacion(self):
        assert extract_artefacts("modificacion") == ["changes"]

    def test_multiple_deduplicated(self):
        a = extract_artefacts("decisiones y decisiones y findings")
        assert a == ["decisions", "findings"]

    def test_empty_for_todo(self):
        # "todo" is not an artefact keyword
        assert extract_artefacts("todo") == []


# ---------------------------------------------------------------------------
# Critical ordering: "todo" must never match cherry-pick patterns
# ---------------------------------------------------------------------------

class TestTodoAlwaysMerge:
    @pytest.mark.parametrize("text", [
        "trae todo de experiment",
        "trae todo desde experiment",
        "trae todo lo de experiment",
        "trae todo lo útil de experiment",
        "dame todo de experiment",
        "dame todo desde feature-x",
        "pásame todo de experiment",
        "pasa todo de feature-x a main",
        "mezcla todo de experiment",
        "mezcla todo lo de feature-x en main",
        "mezcla todo lo útil de feature-x",
        "fusiona todo de experiment",
        "combina todo desde feature-x",
        "integra todo de feature-x",
        "incorpora todo de experiment",
        "incorpora todo lo útil de feature-x a main",
    ])
    def test_is_merge(self, text: str):
        assert intent(text) == "branch_merge", f"Expected merge for: {text!r}"


# ---------------------------------------------------------------------------
# Documented ambiguous cases → conversation (safe fallback)
# ---------------------------------------------------------------------------

class TestAmbiguousFallthrough:
    def test_esta_rama_without_context(self):
        # "esta rama" is a contextual reference; the router has no REPL context
        assert intent("mezcla esta rama en main") == "conversation"

    def test_fusiona_esta_rama_con_main(self):
        assert intent("fusiona esta rama con main") == "conversation"

    def test_pasa_todo_lo_util_de_esta_rama_a_main(self):
        assert intent("pasa todo lo útil de esta rama a main") == "conversation"

    def test_generic_sentence_no_match(self):
        assert intent("revisa el módulo de runtime") == "conversation"

    def test_mid_sentence_merge_word_no_match(self):
        # "merge" mid-sentence should not fire (pattern uses ^)
        assert intent("cuando hago merge de feature no funciona") == "conversation"

    def test_trae_without_recognized_artefact_is_cherry_pick_with_empty_artefacts(self):
        # "trae el código de feature-x" — "codigo" is not a recognized artefact
        # keyword, so extract_artefacts() returns [].  The _cherry_trae pattern
        # still fires (any Spanish word satisfies the word-group part), yielding
        # cherry-pick with artefacts=[].  The REPL handler defaults to
        # decisions+findings in that case — documented safe fallback.
        i, v = route("trae el código de feature-x")
        assert i == "branch_cherry_pick"
        assert v["artefacts"] == []

    def test_incorpora_without_artefact_and_without_branch(self):
        # Single-word branch (≤1 char) violates _BRANCH_NAME — conversation
        assert intent("incorpora x") == "conversation"


# ---------------------------------------------------------------------------
# Branch operations unaffected by the merge/cherry-pick changes
# ---------------------------------------------------------------------------

class TestBranchOpsUnchanged:
    def test_branch_new(self):
        assert intent("crea una rama feature-x") == "branch_new"

    def test_branch_switch(self):
        assert intent("vuelve a main") == "branch_switch"

    def test_branch_list(self):
        assert intent("lista de ramas") == "branch_list"

    def test_branch_show(self):
        assert intent("muestra la rama feature-x") == "branch_show"

    def test_exit(self):
        assert intent("exit") == "exit"

    def test_help(self):
        assert intent("help") == "help"

    def test_status(self):
        assert intent("status") == "session_status"

    def test_slash_commands_pass_through(self):
        assert intent("/branch merge experiment") == "conversation"
        assert intent("/help") == "conversation"


# ---------------------------------------------------------------------------
# Contextual source helpers — is_contextual_source_ref / resolve_contextual_source
# ---------------------------------------------------------------------------

class TestContextualHelpers:
    """Unit tests for the two public helper functions."""

    @pytest.mark.parametrize("text", [
        "esta rama",
        "esta rama activa",
        "la rama actual",
        "la rama activa",
        "la actual",
        "current branch",
        "active branch",
        "this branch",
        "current one",
        "esta",
        # Embedded in a sentence
        "mezcla esta rama en main",
        "trae las decisiones de la actual",
        "fusiona la rama actual con main",
        "combina esta con main",
    ])
    def test_is_contextual_ref_true(self, text: str):
        assert is_contextual_source_ref(text) is True, f"Expected True for: {text!r}"

    @pytest.mark.parametrize("text", [
        "merge experiment",
        "trae las decisiones de feature-x",
        "fusiona feature-x con main",
        "trae todo de experiment",
        "dame los findings de main",
        "crea una rama feature-x",
        "",
    ])
    def test_is_contextual_ref_false(self, text: str):
        assert is_contextual_source_ref(text) is False, f"Expected False for: {text!r}"

    def test_resolve_esta_rama(self):
        result = resolve_contextual_source("mezcla esta rama en main", "feature-x")
        assert result == "mezcla feature-x en main"

    def test_resolve_esta_rama_activa(self):
        result = resolve_contextual_source("fusiona esta rama activa con main", "feature-x")
        assert result == "fusiona feature-x con main"

    def test_resolve_la_rama_actual(self):
        result = resolve_contextual_source("fusiona la rama actual con main", "feature-x")
        assert result == "fusiona feature-x con main"

    def test_resolve_la_actual(self):
        result = resolve_contextual_source("trae las decisiones de la actual", "feature-x")
        assert result == "trae las decisiones de feature-x"

    def test_resolve_la_rama_activa(self):
        result = resolve_contextual_source("incorpora la rama activa en main", "feat-branch")
        assert result == "incorpora feat-branch en main"

    def test_resolve_current_branch(self):
        result = resolve_contextual_source("merge current branch to main", "feat-branch")
        assert result == "merge feat-branch to main"

    def test_resolve_esta_bare(self):
        result = resolve_contextual_source("combina esta con main", "feat-branch")
        assert result == "combina feat-branch con main"

    def test_resolve_only_first_occurrence(self):
        # count=1 replaces only the first match
        result = resolve_contextual_source("mezcla esta rama con esta rama", "feat")
        assert result == "mezcla feat con esta rama"


# ---------------------------------------------------------------------------
# Contextual source merge — route() with active_branch supplied
# ---------------------------------------------------------------------------

class TestContextualSourceMerge:
    """Contextual phrases route to branch_merge when active_branch is provided."""

    BRANCH = "feature-x"

    def _route(self, text: str):
        return route(text, active_branch=self.BRANCH)

    def test_mezcla_esta_rama_en_main(self):
        i, v = self._route("mezcla esta rama en main")
        assert i == "branch_merge"
        assert v["source"] == self.BRANCH

    def test_fusiona_la_rama_actual_con_main(self):
        i, v = self._route("fusiona la rama actual con main")
        assert i == "branch_merge"
        assert v["source"] == self.BRANCH

    def test_haz_merge_de_la_actual_a_main(self):
        i, v = self._route("haz merge de la actual a main")
        assert i == "branch_merge"
        assert v["source"] == self.BRANCH

    def test_incorpora_todo_de_esta_rama_en_main(self):
        i, v = self._route("incorpora todo de esta rama en main")
        assert i == "branch_merge"
        assert v["source"] == self.BRANCH

    def test_combina_esta_con_main(self):
        i, v = self._route("combina esta con main")
        assert i == "branch_merge"
        assert v["source"] == self.BRANCH

    def test_mezcla_esta_en_main(self):
        i, v = self._route("mezcla esta en main")
        assert i == "branch_merge"
        assert v["source"] == self.BRANCH

    def test_pasa_todo_lo_util_de_la_rama_activa_a_main(self):
        i, v = self._route("pasa todo lo útil de la rama activa a main")
        assert i == "branch_merge"
        assert v["source"] == self.BRANCH

    def test_trae_todo_de_esta_rama(self):
        i, v = self._route("trae todo de esta rama")
        assert i == "branch_merge"
        assert v["source"] == self.BRANCH

    def test_trae_todo_de_esta_rama_a_main(self):
        i, v = self._route("trae todo de esta rama a main")
        assert i == "branch_merge"
        assert v["source"] == self.BRANCH

    def test_integra_esta_rama_en_main(self):
        i, v = self._route("integra esta rama en main")
        assert i == "branch_merge"
        assert v["source"] == self.BRANCH

    def test_merge_la_actual(self):
        i, v = self._route("merge la actual")
        assert i == "branch_merge"
        assert v["source"] == self.BRANCH

    def test_absorbe_esta(self):
        i, v = self._route("absorbe esta")
        assert i == "branch_merge"
        assert v["source"] == self.BRANCH

    def test_mezcla_la_rama_activa_en_main(self):
        i, v = self._route("mezcla la rama activa en main")
        assert i == "branch_merge"
        assert v["source"] == self.BRANCH

    def test_fusiona_this_branch_with_main(self):
        i, v = self._route("fusiona this branch con main")
        assert i == "branch_merge"
        assert v["source"] == self.BRANCH

    def test_active_branch_with_dashes(self):
        i, v = route("mezcla esta rama en main", active_branch="my-feature-01")
        assert i == "branch_merge"
        assert v["source"] == "my-feature-01"


# ---------------------------------------------------------------------------
# Contextual source cherry-pick — route() with active_branch supplied
# ---------------------------------------------------------------------------

class TestContextualSourceCherryPick:
    """Contextual phrases route to branch_cherry_pick when an artefact is specified."""

    BRANCH = "feature-x"

    def _route(self, text: str):
        return route(text, active_branch=self.BRANCH)

    def test_trae_solo_las_decisiones_de_esta_rama(self):
        i, v = self._route("trae solo las decisiones de esta rama")
        assert i == "branch_cherry_pick"
        assert v["source"] == self.BRANCH
        assert "decisions" in v["artefacts"]

    def test_trae_solo_las_decisiones_de_esta_rama(self):
        # Cherry-pick patterns do not support a destination suffix ("a main").
        # The destination is always the current active branch.
        i, v = self._route("trae las decisiones de esta rama")
        assert i == "branch_cherry_pick"
        assert v["source"] == self.BRANCH
        assert "decisions" in v["artefacts"]

    def test_pasame_los_findings_de_la_rama_actual(self):
        i, v = self._route("pásame los findings de la rama actual")
        assert i == "branch_cherry_pick"
        assert v["source"] == self.BRANCH
        assert "findings" in v["artefacts"]

    def test_incorpora_unicamente_summary_de_la_actual(self):
        i, v = self._route("incorpora únicamente summary de la actual")
        assert i == "branch_cherry_pick"
        assert v["source"] == self.BRANCH
        assert "summary" in v["artefacts"]

    def test_trae_facts_y_findings_desde_esta_rama(self):
        i, v = self._route("trae facts y findings desde esta rama")
        assert i == "branch_cherry_pick"
        assert v["source"] == self.BRANCH
        assert "facts" in v["artefacts"]
        assert "findings" in v["artefacts"]

    def test_quiero_las_decisiones_de_esta_rama(self):
        i, v = self._route("quiero las decisiones de esta rama")
        assert i == "branch_cherry_pick"
        assert v["source"] == self.BRANCH
        assert "decisions" in v["artefacts"]

    def test_extrae_errores_de_la_rama_activa(self):
        i, v = self._route("extrae los errores de la rama activa")
        assert i == "branch_cherry_pick"
        assert v["source"] == self.BRANCH
        assert "errors" in v["artefacts"]

    def test_cherry_pick_de_la_actual(self):
        # Cherry-pick patterns don't support a destination suffix; "a main" would
        # confuse the parser (captured as trailing artefact text, not a branch).
        i, v = self._route("cherry-pick de la actual")
        assert i == "branch_cherry_pick"
        assert v["source"] == self.BRANCH

    def test_incorpora_solo_decisions_de_la_actual(self):
        i, v = self._route("incorpora solo las decisiones de la actual")
        assert i == "branch_cherry_pick"
        assert v["source"] == self.BRANCH
        assert "decisions" in v["artefacts"]


# ---------------------------------------------------------------------------
# Contextual fallback — without active_branch → conversation
# ---------------------------------------------------------------------------

class TestContextualFallback:
    """Without active_branch, all contextual phrases fall through to conversation."""

    @pytest.mark.parametrize("text", [
        "mezcla esta rama en main",
        "fusiona la rama actual con main",
        "haz merge de la actual a main",
        "trae solo las decisiones de esta rama",
        "pásame los findings de la rama actual",
        "combina esta con main",
        "trae todo de esta rama",
        "incorpora únicamente summary de la actual",
        "cherry-pick de la actual",
        "mezcla esta en main",
        "absorbe esta",
    ])
    def test_no_active_branch_returns_conversation(self, text: str):
        i, v = route(text)  # no active_branch
        assert i == "conversation", f"Expected conversation for: {text!r}"

    @pytest.mark.parametrize("text", [
        "mezcla esta rama en main",
        "fusiona la rama actual con main",
    ])
    def test_empty_string_active_branch_returns_conversation(self, text: str):
        i, v = route(text, active_branch="")
        assert i == "conversation", f"Expected conversation for empty active_branch: {text!r}"


# ---------------------------------------------------------------------------
# Contextual edge / border cases
# ---------------------------------------------------------------------------

class TestContextualEdgeCases:
    """
    Border cases documented with their expected classification and rationale.
    """

    BRANCH = "feature-x"

    def test_mezcla_esto_is_merge_with_esto_as_branch_name(self):
        # "esto" is NOT in _CONTEXTUAL_SOURCE_RE, so no contextual substitution runs.
        # However "esto" satisfies _BRANCH_NAME (≥ 2 alphanumeric chars), so
        # "mezcla esto" is classified as branch_merge with source="esto".
        # Documented: the router cannot distinguish "esto" the pronoun from a
        # branch literally named "esto". Branches with pronoun-like names are
        # discouraged but valid.
        i, v = route("mezcla esto", active_branch=self.BRANCH)
        assert i == "branch_merge"
        assert v["source"] == "esto"

    def test_bare_esta_rama_alone_conversation(self):
        # "esta rama" with no verb → after substitution "feature-x" → no rule → conversation
        i, _ = route("esta rama", active_branch=self.BRANCH)
        assert i == "conversation"

    def test_bare_main_conversation(self):
        # "main" alone → no contextual ref, no rule → conversation
        i, _ = route("main", active_branch=self.BRANCH)
        assert i == "conversation"

    def test_trae_algo_de_esta_rama_cherry_pick_empty_artefacts(self):
        # "algo" is not a recognized artefact keyword → artefacts=[].
        # After substitution: "trae algo de feature-x" → matches _cherry_trae.
        # Documented: cherry-pick with artefacts=[] → REPL defaults to decisions+findings.
        i, v = route("trae algo de esta rama", active_branch=self.BRANCH)
        assert i == "branch_cherry_pick"
        assert v["source"] == self.BRANCH
        assert v["artefacts"] == []

    def test_esta_rama_no_verb_clear_no_match(self):
        # Only the subject, no actionable verb → after substitution no rule fires
        i, _ = route("la rama actual está bien", active_branch=self.BRANCH)
        assert i == "conversation"

    def test_explicit_branch_name_unaffected(self):
        # Explicit branch names must NOT be modified even when active_branch is provided.
        i, v = route("merge experiment", active_branch=self.BRANCH)
        assert i == "branch_merge"
        assert v["source"] == "experiment"  # NOT feature-x

    def test_explicit_cherry_pick_source_unaffected(self):
        i, v = route("trae las decisiones de experiment", active_branch=self.BRANCH)
        assert i == "branch_cherry_pick"
        assert v["source"] == "experiment"  # explicit branch, not active

    def test_slash_commands_unaffected(self):
        i, _ = route("/branch merge experiment", active_branch=self.BRANCH)
        assert i == "conversation"

    def test_regular_conversation_unaffected(self):
        i, _ = route("revisa el módulo de runtime y dime qué falla", active_branch=self.BRANCH)
        assert i == "conversation"


class TestCherryPickWithExplicitDestination:
    """
    Cherry-pick with an explicit destination branch ("a main", "hacia main", "en main").

    All cases: source is a real branch name or a contextual ref resolved to active_branch.
    Destination must appear in result["destination"]; artefacts must be classified correctly.
    """

    ACTIVE = "feature-x"

    # -- explicit source, explicit destination -----------------------------------

    def test_trae_decisions_de_branch_a_main(self):
        i, v = route("trae las decisiones de feature-x a main")
        assert i == "branch_cherry_pick"
        assert v["source"] == "feature-x"
        assert v["destination"] == "main"
        assert "decisions" in v["artefacts"]

    def test_trae_findings_de_branch_hacia_main(self):
        i, v = route("trae los findings de feature-x hacia main")
        assert i == "branch_cherry_pick"
        assert v["source"] == "feature-x"
        assert v["destination"] == "main"
        assert "findings" in v["artefacts"]

    def test_dame_findings_de_branch_en_main(self):
        i, v = route("dame los findings de feature-x en main")
        assert i == "branch_cherry_pick"
        assert v["source"] == "feature-x"
        assert v["destination"] == "main"
        assert "findings" in v["artefacts"]

    def test_pasame_findings_de_branch_a_main(self):
        i, v = route("pásame los findings de feature-x a main")
        assert i == "branch_cherry_pick"
        assert v["source"] == "feature-x"
        assert v["destination"] == "main"
        assert "findings" in v["artefacts"]

    def test_extrae_decisions_y_findings_de_branch_a_main(self):
        i, v = route("extrae las decisiones y findings de feature-x a main")
        assert i == "branch_cherry_pick"
        assert v["source"] == "feature-x"
        assert v["destination"] == "main"
        assert "decisions" in v["artefacts"]
        assert "findings" in v["artefacts"]

    def test_incorpora_summary_de_branch_en_main(self):
        i, v = route("incorpora únicamente el summary de feature-x en main")
        assert i == "branch_cherry_pick"
        assert v["source"] == "feature-x"
        assert v["destination"] == "main"
        assert "summary" in v["artefacts"]

    def test_quiero_facts_de_branch_a_main(self):
        i, v = route("quiero los facts de feature-x a main")
        assert i == "branch_cherry_pick"
        assert v["source"] == "feature-x"
        assert v["destination"] == "main"
        assert "facts" in v["artefacts"]

    def test_importa_decisions_de_branch_a_main(self):
        i, v = route("importa las decisiones de feature-x a main")
        assert i == "branch_cherry_pick"
        assert v["source"] == "feature-x"
        assert v["destination"] == "main"
        assert "decisions" in v["artefacts"]

    def test_cherry_pick_bare_a_main(self):
        i, v = route("cherry-pick feature-x a main")
        assert i == "branch_cherry_pick"
        assert v["source"] == "feature-x"
        assert v["destination"] == "main"

    def test_cherry_pick_de_branch_a_main(self):
        i, v = route("cherry-pick de feature-x a main")
        assert i == "branch_cherry_pick"
        assert v["source"] == "feature-x"
        assert v["destination"] == "main"

    # -- contextual source, explicit destination --------------------------------

    def test_contextual_trae_decisions_de_esta_rama_a_main(self):
        i, v = route("trae las decisiones de esta rama a main", active_branch=self.ACTIVE)
        assert i == "branch_cherry_pick"
        assert v["source"] == self.ACTIVE
        assert v["destination"] == "main"
        assert "decisions" in v["artefacts"]

    def test_contextual_pasame_findings_de_la_actual_a_main(self):
        i, v = route("pásame los findings de la actual a main", active_branch=self.ACTIVE)
        assert i == "branch_cherry_pick"
        assert v["source"] == self.ACTIVE
        assert v["destination"] == "main"
        assert "findings" in v["artefacts"]

    def test_contextual_incorpora_summary_de_esta_rama_en_main(self):
        i, v = route(
            "incorpora únicamente el summary de esta rama en main",
            active_branch=self.ACTIVE,
        )
        assert i == "branch_cherry_pick"
        assert v["source"] == self.ACTIVE
        assert v["destination"] == "main"
        assert "summary" in v["artefacts"]

    def test_contextual_trae_facts_y_findings_desde_esta_rama_hacia_main(self):
        i, v = route(
            "trae facts y findings desde esta rama hacia main",
            active_branch=self.ACTIVE,
        )
        assert i == "branch_cherry_pick"
        assert v["source"] == self.ACTIVE
        assert v["destination"] == "main"
        assert "facts" in v["artefacts"]
        assert "findings" in v["artefacts"]

    def test_contextual_cherry_pick_de_la_rama_actual_decisions_a_main(self):
        # Dest at end so _DEST_BRANCH_RE can anchor to $
        i, v = route(
            "cherry-pick de la rama actual solo decisions a main",
            active_branch=self.ACTIVE,
        )
        assert i == "branch_cherry_pick"
        assert v["source"] == self.ACTIVE
        assert v["destination"] == "main"

    def test_contextual_no_active_branch_falls_to_conversation(self):
        # Without active_branch, contextual refs cannot resolve → conversation
        i, _ = route("trae las decisiones de esta rama a main", active_branch=None)
        assert i == "conversation"

    # -- no destination → destination is None (backward-compat) -----------------

    def test_cherry_pick_without_dest_destination_is_none(self):
        i, v = route("trae las decisiones de feature-x")
        assert i == "branch_cherry_pick"
        assert v["source"] == "feature-x"
        assert v["destination"] is None
        assert "decisions" in v["artefacts"]

    def test_contextual_cherry_pick_without_dest_destination_is_none(self):
        i, v = route("trae las decisiones de esta rama", active_branch=self.ACTIVE)
        assert i == "branch_cherry_pick"
        assert v["source"] == self.ACTIVE
        assert v["destination"] is None
        assert "decisions" in v["artefacts"]

    # -- non-regression: merge contextual still routes to merge ------------------

    def test_merge_contextual_still_merge(self):
        i, v = route("mezcla esta rama en main", active_branch=self.ACTIVE)
        assert i == "branch_merge"
        assert v["source"] == self.ACTIVE
        assert v["destination"] == "main"

    def test_merge_trae_todo_contextual_still_merge(self):
        i, v = route("trae todo de esta rama a main", active_branch=self.ACTIVE)
        assert i == "branch_merge"
        assert v["source"] == self.ACTIVE
        assert v["destination"] == "main"
