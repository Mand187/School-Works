class Fonts:
    SMALL_FONT_STYLE  =  ("Arial", 16)
    LARGE_FONT_STYLE  =  ("Arial", 40, "bold")
    DIGITS_FONT_STYLE =  ("Arial", 24, "bold")
    DEFAULT_FONT_STYLE=  ("Arial", 18)

    @staticmethod
    def get_font(font_name):
        font_value = getattr(Fonts, font_name, None)
        if font_value is None:
            raise ValueError(f"Invalid font name: {font_name}")
        return font_value