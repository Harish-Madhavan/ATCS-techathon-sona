import unittest


class TestThemeHelpers(unittest.TestCase):
    def test_truncate(self):
        from gui.theme import truncate_middle
        self.assertEqual(truncate_middle("abc", 10), "abc")
        short = truncate_middle("x" * 50, 20)
        self.assertEqual(len(short), 20)
        self.assertIn("...", short)

    def test_palette_constants(self):
        import gui.theme as t
        for name in ("BG_DARK", "BG_CARD", "FG_LIGHT", "ACCENT", "GREEN", "RED", "YELLOW"):
            val = getattr(t, name)
            self.assertTrue(val.startswith("#"), name)

    def test_badge_kind_switch_without_display(self):
        # set_badge_kind must tolerate duck-typed widgets (no Tk needed).
        from gui.theme import set_badge_kind
        seen = {}

        class Fake:
            def config(self, **kw):
                seen.update(kw)
        set_badge_kind(Fake(), "emergency")
        self.assertEqual(seen.get("bg"), "#FF453A")


if __name__ == "__main__":
    unittest.main()
