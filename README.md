PixelCanvas.io downloader
=========================

This utility lets you download the http://pixelcanvas.io bitmap and export regions of it to .png files.

# Requirements

    pip install -r requirements.txt

# Usage

This tool is run from the command line, where you provide the coordinates you
want to download and render.

The format for typing coordinates is `UPPERLEFT--LOWERRIGHT`. The format for
each of those pieces is `X.Y`.

Sometimes, argparse gets confused by negative coordinates because it thinks
you're trying to provide another argument. Sorry.
If this happens, use a tilde `~` as the negative sign instead.

Remember, because this is an image, up and left are negative;
down and right are positive.

Commands:

    update:
        Download chunks into the database.

        > pixelcanvas.py update ~100.~100--100.100

    render:
        Export an image as PNG.

        > pixelcanvas.py render 0.0--100.100 <flags>

        flags:
        --show:
            Instead of saving the image, display it on the screen.
            https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.show

        --update:
            Update the chunks before exporting them.

So, for example:

    > pixelcanvas.py update 0.0--100.100
    > pixelcanvas.py update ~100.~100--100.100
    > pixelcanvas.py update ~1200.300--~900.600

    > pixelcanvas.py render 0.0--100.100
    > pixelcanvas.py render ~100.~100--100.100 --update
    > pixelcanvas.py render ~1200.300--~900.600 --show

# To do

Here are some things we might like to improve:

- Some way to get a statistics overview or visual map of which chunks we have in the database, so we know what we're missing.
- Render the image is it appeared at some point in the past, taking advantage of the `updated_at` column.
- Probably never going to happen: A GUI application to browse the db just like the site.
