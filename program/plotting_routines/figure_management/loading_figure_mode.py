

def loading_figure_mode(working_mode):
    if working_mode is "development":
        from figure_management import figure_devolpment as fma
    else:
        from figure_management import figure_management as fma
    return fma
