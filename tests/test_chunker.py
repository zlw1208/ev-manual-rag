from app.ingestion.chunker import PageRecord, create_chunks, reconstruct_paragraphs
from app.ingestion.sources import ManualSource

SOURCE = ManualSource(
    id="demo",
    brand="AITO",
    model="M5",
    power_type="纯电",
    document_type="使用说明书",
    source_url="https://example.com/manual.pdf",
    local_filename="demo.pdf",
)


def make_page(page_number: int, heading: str, text: str) -> PageRecord:
    return PageRecord(
        document_id="demo",
        page_number=page_number,
        headings=[heading],
        text=text,
        safety_markers=[],
        extraction_mode="single",
    )


def test_reconstruct_paragraphs_joins_visual_line_wraps() -> None:
    text = "充电说明\n充电前请确认充电接口内没有\n水或异物。\n● 请连接充电枪。"

    paragraphs = reconstruct_paragraphs(text, ["充电说明"])

    assert paragraphs == ["充电前请确认充电接口内没有水或异物。", "● 请连接充电枪。"]


def test_chunks_do_not_cross_section_boundaries() -> None:
    pages = [
        make_page(1, "充电说明", "充电说明\n充电前检查接口。"),
        make_page(2, "动力电池", "动力电池\n请定期检查动力电池状态。"),
    ]

    chunks = list(
        create_chunks(
            pages,
            SOURCE,
            target_chars=30,
            min_chars=5,
            max_chars=60,
            overlap_chars=10,
        )
    )

    assert len(chunks) == 2
    assert chunks[0].section_path == ["充电说明"]
    assert chunks[1].section_path == ["动力电池"]


def test_table_of_contents_is_not_searchable() -> None:
    pages = [make_page(5, "目录", "目录\n充电说明........156\n动力电池........165")]

    chunk = next(create_chunks(pages, SOURCE, target_chars=50, min_chars=5, max_chars=100))

    assert chunk.searchable is False
    assert chunk.chunk_type == "table_of_contents"


def test_long_text_is_split_under_hard_limit() -> None:
    pages = [make_page(10, "长章节", "长章节\n" + "充电注意事项。" * 80)]

    chunks = list(
        create_chunks(
            pages,
            SOURCE,
            target_chars=120,
            min_chars=50,
            max_chars=180,
            overlap_chars=20,
        )
    )

    assert len(chunks) > 1
    assert all(chunk.char_count <= 180 for chunk in chunks)
    assert all(chunk.section_path == ["长章节"] for chunk in chunks)


def test_safety_detection_uses_marker_position_not_any_substring() -> None:
    pages = [
        make_page(1, "安全驾驶", "安全驾驶\n驾驶时请集中注意力。"),
        make_page(2, "充电安全", "充电安全\n警告\n请勿拆卸充电设备。"),
    ]

    chunks = list(
        create_chunks(
            pages,
            SOURCE,
            target_chars=30,
            min_chars=5,
            max_chars=80,
            overlap_chars=10,
        )
    )

    assert chunks[0].safety_level == "normal"
    assert chunks[1].safety_level == "warning"


def test_inline_subheadings_create_retrieval_boundaries() -> None:
    pages = [
        make_page(
            10,
            "充电说明",
            "充电说明\n充电安全警告\n警告\n请勿拆卸充电设备。\n充电方式\n车辆支持交流充电。",
        )
    ]

    chunks = list(
        create_chunks(
            pages,
            SOURCE,
            target_chars=50,
            min_chars=5,
            max_chars=100,
            overlap_chars=10,
        )
    )

    assert ["充电说明", "充电安全警告"] in [chunk.section_path for chunk in chunks]
    assert ["充电说明", "充电方式"] in [chunk.section_path for chunk in chunks]
