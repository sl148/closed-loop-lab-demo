"""Archive ingest, BM25 search, page reads and quote verification (kb/archive.py)."""

import yaml

from labloop.kb.archive import Archive, ingest


def make_kb(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.txt").write_text(
        "Cell culture basics. Work with one cell line at a time and clean the hood between lines.\f"
        "Viability assay. Equilibrate the plate to room temperature for approximately 30 minutes "
        "before adding reagent, then mix for 2 minutes.")
    (tmp_path / "docs" / "sheet.html").write_text(
        "<html><body><p>Aqueous solutions should not be stored for more than one day.</p></body></html>")
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump({"docs": [
        {"id": "guide", "title": "Guide", "file": "guide.txt"},
        {"id": "sheet", "title": "Datasheet", "file": "sheet.html"}]}))
    assert ingest(tmp_path) == 3
    return Archive(tmp_path)


def test_search_read_verify(tmp_path):
    kb = make_kb(tmp_path)
    hit = kb.search("equilibrate plate room temperature minutes before reagent")[0]
    assert (hit["doc"], hit["page"]) == ("guide", 2)
    assert kb.search("aqueous storage one day")[0]["doc"] == "sheet"
    assert kb.read_pages("guide", 1, 2)[1]["page"] == 2
    assert kb.verify_quote("guide", 2, "room temperature for approximately 30 minutes")
    assert not kb.verify_quote("guide", 2, "room temperature for approximately 60 minutes")  # wrong number
    assert not kb.verify_quote("guide", 1, "room temperature for approximately 30 minutes")  # wrong page
    assert not kb.verify_quote("guide", 2, "30 minutes")  # too short to pin anything down
