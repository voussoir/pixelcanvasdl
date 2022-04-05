import argparse
import datetime
import gzip
import PIL.Image
import random
import requests
import sqlite3
import sys
import time

from voussoirkit import threadpool
from voussoirkit import vlogging

log = vlogging.getLogger(__name__, 'pixelcanvasdl')
vlogging.getLogger('urllib3.connectionpool').setLevel(vlogging.CRITICAL)

WHITE = (255, 255, 255)
LIGHTGRAY = (228, 228, 228)
DARKGRAY = (136, 136, 136)
BLACK = (34, 34, 34)
PINK = (255, 167, 209)
RED = (229, 0, 0)
ORANGE = (229, 149, 0)
BROWN = (160, 106, 66)
YELLOW = (229, 217, 0)
LIGHTGREEN = (148, 224, 68)
DARKGREEN = (2, 190, 1)
LIGHTBLUE = (0, 211, 221)
MEDIUMBLUE = (0, 131, 199)
DARKBLUE = (0, 0, 234)
LIGHTPURPLE = (207, 110, 228)
DARKPURPLE = (130, 0, 128)

COLOR_MAP = {
     0: WHITE,
     1: LIGHTGRAY,
     2: DARKGRAY,
     3: BLACK,
     4: PINK,
     5: RED,
     6: ORANGE,
     7: BROWN,
     8: YELLOW,
     9: LIGHTGREEN,
    10: DARKGREEN,
    11: LIGHTBLUE,
    12: MEDIUMBLUE,
    13: DARKBLUE,
    14: LIGHTPURPLE,
    15: DARKPURPLE,
}

# The width and height of a chunk, in pixels.
CHUNK_SIZE_PIX = 64

# The number of bytes for a full chunk.
# They are 32x64 because each byte represents two 4-bit pixels.
CHUNK_SIZE_BYTES = int(CHUNK_SIZE_PIX * (CHUNK_SIZE_PIX / 2))

# The width and height of a bigchunk, in chunks.
BIGCHUNK_SIZE_CHUNKS = 15

# The width and height of a bigchunk, in pixels.
BIGCHUNK_SIZE_PIX = BIGCHUNK_SIZE_CHUNKS * CHUNK_SIZE_PIX

# The number of bytes for a full bigchunk.
BIGCHUNK_SIZE_BYTES = int(BIGCHUNK_SIZE_PIX * (BIGCHUNK_SIZE_PIX / 2))

# The chunk 0, 0 has a pixel coordinate of -448, -448 for some reason.
ORIGIN_OFFSET_X = 448
ORIGIN_OFFSET_Y = 448

DB_INIT = '''
BEGIN;
CREATE TABLE IF NOT EXISTS chunks (x INT, y INT, data BLOB, updated_at REAL);
CREATE INDEX IF NOT EXISTS chunks_x_y ON chunks(x, y);
COMMIT;
'''

sql = sqlite3.connect('pixelcanvas.db')
cur = sql.cursor()
cur.executescript(DB_INIT)

session = requests.Session()

# HELPER FUNCTIONS
################################################################################
def now():
    n = datetime.datetime.now(datetime.timezone.utc)
    return n.timestamp()

# DB FUNCTIONS
################################################################################
def db_commit():
    log.debug('Committing')
    sql.commit()

def get_chunk_from_db(chunk_x, chunk_y, as_of=None):
    '''
    Get the chunk from the database, and raise IndexError if it doesn't exist.
    '''
    query = f'''
    SELECT x, y, data FROM chunks
    WHERE x == ? AND y == ?
    {'AND updated_at <= ?' if as_of is not None else ''}
    ORDER BY updated_at DESC
    LIMIT 1
    '''
    bindings = [chunk_x, chunk_y]
    if as_of is not None:
        bindings.append(as_of)
    cur.execute(query, bindings)
    fetch = cur.fetchone()
    if fetch is None:
        raise IndexError(f'{chunk_x}, {chunk_y}')
    (x, y, data) = fetch
    data = gzip.decompress(data)
    return (x, y, data)

def get_chunk(chunk_x, chunk_y, *args, **kwargs):
    '''
    Get the chunk from the database if it exists, or else download it.
    '''
    try:
        return get_chunk_from_db(chunk_x, chunk_y, *args, **kwargs)
    except IndexError:
        (bigchunk_x, bigchunk_y) = chunk_to_bigchunk(chunk_x, chunk_y)
        chunks = download_bigchunk(bigchunk_x, bigchunk_y)
        insert_chunks(chunks)
        return get_chunk_from_db(chunk_x, chunk_y, *args, **kwargs)

def insert_chunk(chunk_x, chunk_y, data, commit=True):
    try:
        existing_chunk = get_chunk_from_db(chunk_x, chunk_y)
    except IndexError:
        pass
    else:
        if data == existing_chunk[2]:
            return
    log.debug('Updating chunk %s %s', chunk_x, chunk_y)
    data = gzip.compress(data)
    cur.execute('INSERT INTO chunks VALUES(?, ?, ?, ?)', [chunk_x, chunk_y, data, now()])
    if commit:
        db_commit()

def insert_chunks(chunks, commit=True):
    for (index, chunk) in enumerate(chunks):
        if index % 25000 == 0 and commit:
            db_commit()
        insert_chunk(*chunk, commit=False)
    if commit:
        db_commit()

# API FUNCTIONS
################################################################################
def url_for_bigchunk(bigchunk_x, bigchunk_y):
    return f'http://pixelcanvas.io/api/bigchunk/{bigchunk_x}.{bigchunk_y}.bmp'

def request(url):
    response = session.get(url)
    response.raise_for_status()
    return response

def download_bigchunk(bigchunk_x, bigchunk_y):
    '''
    Download a bigchunk and return the list of chunks.
    '''
    url = url_for_bigchunk(bigchunk_x, bigchunk_y)
    log.info('Downloading %s', url)
    response = request(url)
    bigchunk_data = response.content
    if len(bigchunk_data) != BIGCHUNK_SIZE_BYTES:
        message = 'Received bigchunk does not matched the expected byte size!\n'
        message += 'Got %d instead of %d' % (len(bigchunk_data), BIGCHUNK_SIZE_BYTES)
        raise ValueError(message)
    chunks = split_bigchunk(bigchunk_x, bigchunk_y, bigchunk_data)
    return chunks

def download_bigchunk_range(bigchunk_xy1, bigchunk_xy2, shuffle=False, threads=1):
    '''
    Given (UPPERLEFT_X, UPPERLEFT_Y), (LOWERRIGHT_X, LOWERRIGHT_Y),
    download multiple bigchunks, and yield all of the small chunks.
    '''
    if threads < 1:
        raise ValueError(threads)

    log.debug('Downloading bigchunk range %s-%s', bigchunk_xy1, bigchunk_xy2)
    bigchunks = bigchunk_range_iterator(bigchunk_xy1, bigchunk_xy2)
    if shuffle:
        bigchunks = list(bigchunks)
        random.shuffle(bigchunks)

    if threads == 1:
        for (x, y) in bigchunks:
            chunks = download_bigchunk(x, y)
            yield from chunks

    else:
        pool = threadpool.ThreadPool(size=threads)
        job_gen = (
            {'function': download_bigchunk, 'args': (x, y), 'name': (x, y),}
            for (x, y) in bigchunks
        )
        pool.add_generator(job_gen)
        for job in pool.result_generator():
            job.join()
            if job.exception:
                raise job.exception
            yield from job.value

# CHUNK FUNCTIONS
################################################################################
def chunk_range_iterator(chunk_xy1, chunk_xy2):
    '''
    Given (UPPERLEFT_X, UPPERLEFT_Y), (LOWERRIGHT_X, LOWERRIGHT_Y),
    yield (x, y) pairs for chunks in this range, inclusive.
    '''
    for x in range(chunk_xy1[0], chunk_xy2[0] + 1):
        for y in range(chunk_xy1[1], chunk_xy2[1] + 1):
            yield (x, y)

def bigchunk_range_iterator(bigchunk_xy1, bigchunk_xy2):
    '''
    Given (UPPERLEFT_X, UPPERLEFT_Y), (LOWERRIGHT_X, LOWERRIGHT_Y),
    yield (x, y) pairs for bigchunks in this range, inclusive.
    '''
    for x in range(bigchunk_xy1[0], bigchunk_xy2[0] + BIGCHUNK_SIZE_CHUNKS, BIGCHUNK_SIZE_CHUNKS):
        for y in range(bigchunk_xy1[1], bigchunk_xy2[1] + BIGCHUNK_SIZE_CHUNKS, BIGCHUNK_SIZE_CHUNKS):
            yield (x, y)

def chunk_to_bigchunk(chunk_x, chunk_y):
    bigchunk_x = (chunk_x // BIGCHUNK_SIZE_CHUNKS) * BIGCHUNK_SIZE_CHUNKS
    bigchunk_y = (chunk_y // BIGCHUNK_SIZE_CHUNKS) * BIGCHUNK_SIZE_CHUNKS
    log.loud('Converted chunk %s, %s to bigchunk %s, %s', chunk_x, chunk_y, bigchunk_x, bigchunk_y)
    return (bigchunk_x, bigchunk_y)

def chunk_range_to_bigchunk_range(chunk_xy1, chunk_xy2):
    bigchunk_range = (chunk_to_bigchunk(*chunk_xy1), chunk_to_bigchunk(*chunk_xy2))
    return bigchunk_range

def chunk_to_pixel(chunk_x, chunk_y):
    pixel_x = chunk_x * CHUNK_SIZE_PIX - ORIGIN_OFFSET_X
    pixel_y = chunk_y * CHUNK_SIZE_PIX - ORIGIN_OFFSET_Y
    log.loud('Converted chunk %s, %s to pixel %s, %s', chunk_x, chunk_y, pixel_x, pixel_y)
    return (pixel_x, pixel_y)

def chunk_range_to_pixel_range(chunk_xy1, chunk_xy2):
    pixel_range = (chunk_to_pixel(*chunk_xy1), chunk_to_pixel(*chunk_xy2))
    return pixel_range

def pixel_to_chunk(pixel_x, pixel_y):
    chunk_x = (pixel_x + ORIGIN_OFFSET_X) // CHUNK_SIZE_PIX
    chunk_y = (pixel_y + ORIGIN_OFFSET_Y) // CHUNK_SIZE_PIX
    log.loud('Converted pixel %s, %s to chunk %s, %s', pixel_x, pixel_y, chunk_x, chunk_y)
    return (chunk_x, chunk_y)

def pixel_range_to_chunk_range(pixel_xy1, pixel_xy2):
    chunk_range = (pixel_to_chunk(*pixel_xy1), pixel_to_chunk(*pixel_xy2))
    log.loud('Converted pixel range %s, %s to chunk range %s, %s', pixel_xy1, pixel_xy2, *chunk_range)
    return chunk_range

def pixel_to_bigchunk(pixel_x, pixel_y):
    bigchunk_x = ((pixel_x + ORIGIN_OFFSET_X) // BIGCHUNK_SIZE_PIX) * BIGCHUNK_SIZE_CHUNKS
    bigchunk_y = ((pixel_y + ORIGIN_OFFSET_Y) // BIGCHUNK_SIZE_PIX) * BIGCHUNK_SIZE_CHUNKS
    log.loud('Converted pixel %s, %s to bigchunk %s, %s', pixel_x, pixel_y, bigchunk_x, bigchunk_y)
    return (bigchunk_x, bigchunk_y)

def pixel_range_to_bigchunk_range(pixel_xy1, pixel_xy2):
    bigchunk_range = (pixel_to_bigchunk(*pixel_xy1), pixel_to_bigchunk(*pixel_xy2))
    log.loud('Converted pixel range %s, %s to bigchunk range %s, %s', pixel_xy1, pixel_xy2, *bigchunk_range)
    return bigchunk_range

def split_bigchunk(bigchunk_x, bigchunk_y, bigchunk_data):
    '''
    Chunks are downloaded from the site as a "bigchunk" which is just 15x15
    chunks stitched together.
    The chunks are arranged left to right, top to bottom.
    For example, the byte stream:
        000011112222333344445555666677778888
    represents the bitmap:
        001122
        001122
        334455
        334455
        667788
        667788
    '''
    chunks = []
    chunk_count = int(len(bigchunk_data) / CHUNK_SIZE_BYTES)
    for chunk_index in range(chunk_count):
        chunk_x = (chunk_index % BIGCHUNK_SIZE_CHUNKS) + bigchunk_x
        chunk_y = (chunk_index // BIGCHUNK_SIZE_CHUNKS) + bigchunk_y
        start_index = chunk_index * CHUNK_SIZE_BYTES
        end_index = start_index + CHUNK_SIZE_BYTES
        chunk_data = bigchunk_data[start_index:end_index]
        chunk = (chunk_x, chunk_y, chunk_data)
        chunks.append(chunk)
    return chunks

# IMAGE FUNCTIONS
################################################################################
def chunk_to_rgb(chunk_data):
    '''
    Convert the data chunk into RGB tuples.

    PixelCanvas chunks are strings of bytes where every byte represents two
    horizontal pixels. Each pixel is 4 bits since there are 16 colors.
    Chunks are 32x64 bytes for a total of 64x64 pixels.
    '''
    # Each byte actually represents two horizontal pixels. 8F is actually 8, F.
    # So create a generator that takes in the bytes and yields the pixel bits.
    pixels = (
        pixel 
        for byte in chunk_data
        for pixel in (byte >> 4, byte & 0xf)
    )

    matrix = [None for x in range(len(chunk_data) * 2)]
    for (index, pixel) in enumerate(pixels):
        px = index % CHUNK_SIZE_PIX
        py = index // CHUNK_SIZE_PIX
        matrix[(py * CHUNK_SIZE_PIX) + px] = COLOR_MAP[pixel]
    return matrix

def rgb_to_image(matrix):
    matrix = bytes([color for pixel in matrix for color in pixel])
    i = PIL.Image.frombytes(mode='RGB', size=(CHUNK_SIZE_PIX, CHUNK_SIZE_PIX), data=matrix)
    return i

def chunk_to_image(chunk_data, scale=1):
    image = rgb_to_image(chunk_to_rgb(chunk_data))
    if scale is not None and scale != 1:
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, resample=PIL.Image.NEAREST)
    return image

def chunks_to_image(chunks, scale=1):
    '''
    Combine all of the given chunks into a single image.
    '''
    log.debug('Creating image from %s chunks', len(chunks))
    min_x = min(chunk[0] for chunk in chunks)
    max_x = max(chunk[0] for chunk in chunks)
    min_y = min(chunk[1] for chunk in chunks)
    max_y = max(chunk[1] for chunk in chunks)
    span_x = max_x - min_x + 1
    span_y = max_y - min_y + 1
    img_width = int(span_x * CHUNK_SIZE_PIX * scale)
    img_height = int(span_y * CHUNK_SIZE_PIX * scale)
    img = PIL.Image.new(mode='RGB', size=(img_width, img_height), color=WHITE)
    for (chunk_x, chunk_y, chunk_data) in chunks:
        paste_x = int((chunk_x - min_x) * CHUNK_SIZE_PIX * scale)
        paste_y = int((chunk_y - min_y) * CHUNK_SIZE_PIX * scale)
        chunk_image = chunk_to_image(chunk_data, scale)
        img.paste(chunk_image, (paste_x, paste_y))
    return img

def crop_image(image, pixel_xy1, pixel_xy2):
    '''
    Because the images are rendered on a chunk basis, they are probably larger
    than the exact area that you want. Use this function to crop the image to
    the exact coordinates.
    pixel_xy1 and pixel_xy2 are the world coordinates that you used to get this
    image in the first place, not coordinates within this picture.
    '''
    img_width = pixel_xy2[0] - pixel_xy1[0] + 1
    img_height = pixel_xy2[1] - pixel_xy1[1] + 1
    basis_xy = chunk_to_pixel(*pixel_to_chunk(*pixel_xy1))

    xy1 = (pixel_xy1[0] - (basis_xy[0]), pixel_xy1[1] - (basis_xy[1]))
    xy2 = (xy1[0] + img_width, xy1[1] + img_height)
    bbox = (xy1[0], xy1[1], xy2[0], xy2[1])
    log.debug('Cropping image down to %s', bbox)
    image = image.crop(bbox)
    return image

# COMMAND LINE
################################################################################
from voussoirkit import betterhelp

def parse_coordinate_string(coordinates):
    '''
    Convert the given '~100.~100--100.100' to ((-100, -100), (100, 100)).
    '''
    coordinates = coordinates.strip()
    if '--' in coordinates:
        (xy1, xy2) = coordinates.split('--', 1)
    else:
        # If you are only updating a single chunk.
        xy1 = coordinates
        xy2 = coordinates

    def split_xy(xy):
        xy = xy.replace('~', '-')
        xy = xy.replace(',', '.')
        (x, y) = xy.split('.')
        return (int(x), int(y))

    (xy1, xy2) = (split_xy(xy1), split_xy(xy2))
    log.debug('Parsed coordinates %s into %s %s', coordinates, xy1, xy2)
    return (xy1, xy2)

def overview_argparse(args):
    cur.execute('SELECT x, y, updated_at FROM chunks GROUP BY x, y ORDER BY updated_at DESC')
    chunks = cur.fetchall()
    min_x = min(chunk[0] for chunk in chunks)
    max_x = max(chunk[0] for chunk in chunks)
    min_y = min(chunk[1] for chunk in chunks)
    max_y = max(chunk[1] for chunk in chunks)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    x_offset = abs(min(min_x, 0))
    y_offset = abs(min(min_y, 0))

    matrix = [[' ' for x in range(width)] for y in range(height)]
    for (x, y, updated_at) in chunks:
        x += x_offset
        y += y_offset
        matrix[y][x] = '.'

    for (x, y, updated_at) in chunks:
        if (x % 15 == 0) and (y % 15 == 0):
            text = f'{x},{y}'
            x += x_offset
            y += y_offset
            for c in text:
                matrix[y][x] = c
                x+=1

    for row in matrix:
        for column in row:
            print(column, end='')
        print()

def render_argparse(args):
    coordinates = parse_coordinate_string(args.coordinates)
    if args.is_chunks:
        chunk_range = coordinates
        coordinates = chunk_range_to_pixel_range(*coordinates)
    else:
        chunk_range = pixel_range_to_chunk_range(*coordinates)

    chunks = [get_chunk(*chunk_xy) for chunk_xy in chunk_range_iterator(*chunk_range)]
    image = chunks_to_image(chunks, scale=args.scale)

    if args.scale == 1 and not args.is_chunks:
        image = crop_image(image, *coordinates)

    if args.do_show:
        image.show()
    else:
        ((p1x, p1y), (p2x, p2y)) = coordinates
        scale_s = f'_{args.scale}' if args.scale != 1 else ''
        filename = f'{p1x}.{p1y}--{p2x}.{p2y}{scale_s}.png'
        image.save(filename)
        log.debug('Wrote %s', filename)

def update_argparse(args):
    coordinates = parse_coordinate_string(args.coordinates)
    if args.is_chunks:
        bigchunk_range = chunk_range_to_bigchunk_range(*coordinates)
    else:
        bigchunk_range = pixel_range_to_bigchunk_range(*coordinates)
    chunks = download_bigchunk_range(*bigchunk_range, shuffle=args.shuffle, threads=args.threads)
    try:
        insert_chunks(chunks)
    except KeyboardInterrupt:
        db_commit()

@vlogging.main_decorator
def main(argv):
    parser = argparse.ArgumentParser(
        description='''
        This tool is run from the command line, where you provide the coordinates you
        want to download and render.

        The format for coordinates is `UPPERLEFT--LOWERRIGHT`. The format for each
        of those pieces is `X.Y`.

        Sometimes, argparse gets confused by negative coordinates because it thinks
        you're trying to provide another argument. If this happens, use a tilde `~`
        as the negative sign instead.

        Remember, because this is an image, up and left are negative; down and right
        are positive.
        ''',
    )
    subparsers = parser.add_subparsers()

    ################################################################################################

    p_update = subparsers.add_parser(
        'update',
        description='''
        Download chunks into the database.
        ''',
    )
    p_update.examples = [
        '0.0--100.100',
        '~100.~100--100.100',
        '~1200.300--~900.600',
    ]
    p_update.add_argument(
        'coordinates',
    )
    p_update.add_argument(
        '--chunks',
        dest='is_chunks',
        action='store_true',
        help='''
        The coordinates which you provided are chunk coordinates instead of
        pixel coordinates.
        ''',
    )
    p_update.add_argument(
        '--shuffle',
        action='store_true',
        help='''
        Download chunks in a random order instead of from corner to corner.
        ''',
    )
    p_update.add_argument(
        '--threads',
        type=int,
        default=1,
        help='''
        Use X threads to download bigchunks.
        ''',
    )
    p_update.set_defaults(func=update_argparse)

    ################################################################################################

    p_render = subparsers.add_parser(
        'render',
        description='''
        Export an image as PNG.
        ''',
    )
    p_render.examples = [
        '0.0--100.100',
        '~100.~100--100.100 --scale 2',
        '~1200.300--~900.600 --show',
    ]
    p_render.add_argument(
        'coordinates',
    )
    p_render.add_argument(
        '--chunks',
        dest='is_chunks',
        action='store_true',
        help='''
        The coordinates which you provided are chunk coordinates instead of
        pixel coordinates.
        ''',
    )
    p_render.add_argument(
        '--show',
        dest='do_show',
        action='store_true',
        help='''
        Render the image at a different scale.
        For best results, use powers of 2 like 0.5, 0.25, etc.
        This will disable the autocropping.
        ''',
    )
    p_render.add_argument(
        '--scale',
        dest='scale',
        type=float,
        default=1,
        help='''
        Instead of saving the image, display it on the screen.
        pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.show
        ''',
    )
    p_render.set_defaults(func=render_argparse)

    ################################################################################################

    p_overview = subparsers.add_parser(
        'overview',
        help='''
        Draw an ascii map representing the owned chunks.
        ''',
    )
    p_overview.set_defaults(func=overview_argparse)

    return betterhelp.go(parser, argv)

if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
