class Color:
    LIGHT_GRAY = "#F5F5F5"
    LIGHT_BLUE = "#CCEDFF"
    LABEL_COLOR = "#25265E"
    WHITE = "#FFFFFF"
    OFF_WHITE = "#F8FAFF"

    @staticmethod
    def get_color(color_name):
        color_value = getattr(Color, color_name, None)
        if color_value is None:
            raise ValueError(f"Invalid color name: {color_name}")
        return color_value