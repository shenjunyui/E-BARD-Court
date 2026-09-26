import unittest

from court_grounding.src.schema import (
    extract_json,
    parse_prediction,
    parse_prediction_lenient,
    validate_prediction,
)
from court_grounding.src.infer_deepseek import parse_visible_labels


class SchemaTests(unittest.TestCase):
    def test_extracts_fenced_json(self):
        value = extract_json('answer:\n```json\n{"court_lines": []}\n```')
        self.assertEqual(value, {"court_lines": []})

    def test_accepts_valid_line(self):
        value = validate_prediction({
            "court_lines": [{
                "label": "baseline",
                "type": "line",
                "points": [[0, 5], [99, 20]],
            }]
        }, width=100, height=50)
        self.assertEqual(value["court_lines"][0]["points"][1], [99.0, 20.0])

    def test_rejects_unknown_label(self):
        with self.assertRaisesRegex(ValueError, "label is invalid"):
            validate_prediction({
                "court_lines": [{"label": "logo", "type": "line", "points": [[1, 2], [3, 4]]}]
            })

    def test_rejects_out_of_bounds_coordinate(self):
        with self.assertRaisesRegex(ValueError, "outside image width"):
            parse_prediction(
                '{"court_lines":[{"label":"sideline","type":"line","points":[[1,2],[100,4]]}]}',
                width=100,
                height=50,
            )

    def test_lenient_parser_clips_and_skips_individual_markings(self):
        prediction, warnings = parse_prediction_lenient(
            '{"court_lines":['
            '{"label":"baseline","type":"line","points":[[0,10],[104,10]]},'
            '{"label":"center_circle","type":"circle","points":[[50,25]]}'
            ']}',
            width=100,
            height=50,
        )
        self.assertEqual(len(prediction["court_lines"]), 1)
        self.assertEqual(prediction["court_lines"][0]["points"][1], [99.0, 10.0])
        self.assertEqual(len(warnings), 2)

    def test_deepseek_presence_parser_deduplicates_and_orders(self):
        labels = parse_visible_labels(
            '```json\n{"visible_labels":["center_circle","half_court_line",'
            '"center_circle"]}\n```'
        )
        self.assertEqual(labels, ["half_court_line", "center_circle"])

    def test_deepseek_presence_parser_rejects_unknown_label(self):
        with self.assertRaisesRegex(ValueError, "invalid visible label"):
            parse_visible_labels('{"visible_labels":["logo"]}')


if __name__ == "__main__":
    unittest.main()
