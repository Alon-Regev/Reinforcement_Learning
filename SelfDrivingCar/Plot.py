import matplotlib
matplotlib.use("Agg")

import matplotlib.backends.backend_agg as agg

import pylab

def plot_to_image(data):
    """
    plots data and returns an image as bytes
    """
    fig = pylab.figure(figsize=[4.75, 4], # Inches
                    dpi=80,        # dots per inch 
                    )
    ax = fig.gca()
    # set style
    ax.set_xlabel("Generation")
    ax.set_ylabel("Score")
    ax.set_title("Scores")
    ax.grid(True, which='both')

    ax.plot(data)

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    # invert colors (black background)
    raw_data = bytes(map(lambda x: 255 - x, raw_data))

    # close fig
    pylab.close(fig)

    return raw_data, (int(fig.get_figwidth() * fig.get_dpi()), int(fig.get_figheight() * fig.get_dpi()))
