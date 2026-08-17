from epistemic_geometry.benchmarks.e3.rendering import render_latent, template_hashes
from epistemic_geometry.benchmarks.e3.splits import generate_latent


def test_views_share_latent_identity_but_have_distinct_surface_channel_ids() -> None:
    item = generate_latent("MODREG10", "depth_8", 11)
    decimal = render_latent(item)
    twin = render_latent(item, surface="surface_twin")
    word = render_latent(item, response_channel="number_word")
    assert decimal.latent_id == twin.latent_id == word.latent_id
    assert decimal.target == twin.target == word.target == item.target
    assert decimal.view_id != twin.view_id != word.view_id
    assert decimal.prompt != twin.prompt
    assert decimal.prompt != word.prompt
    assert decimal.prompt_hash != twin.prompt_hash
    assert word.target_text.isalpha()


def test_templates_are_frozen_and_complete() -> None:
    hashes = template_hashes()
    assert len(hashes) == 16
    assert all(len(value) == 64 for value in hashes.values())
