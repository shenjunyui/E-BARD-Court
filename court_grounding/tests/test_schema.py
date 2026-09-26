import unittest

from court_grounding.src.schema import extract_json, parse_prediction, validate_prediction


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


if __name__ == "__main__":
    unittest.main()

