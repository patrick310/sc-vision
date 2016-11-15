import picamera # PiCamera()
import io #BytesIO()


MAX_CAM_RESOLUTION = (3280, 2464)
DEFAULT_EXPOSURE_MODE = 'auto'
DEFAULT_FILE_TYPE = 'jpeg'

# =========================================================
# Captures image to stream and returns the stream.
# Params:
# - stream          => io.BytesIO
# - resolution      => 2 integer tuple (x, y)
# - exposure_mode   => string
# - filetype        => string
# Returns:          io.BytesIO
# =========================================================
def capture_to_stream(
                    stream = None,
                    resolution = None,
                    exposure_mode = None,
                    filetype = None
                    ):
    if stream is None:
        stream = io.BytesIO()
    if resolution is None:
        resolution = MAX_CAM_RESOLUTION
    if exposure_mode is None:
        exposure_mode = DEFAULT_EXPOSURE_MODE
    if filetype is None:
        filetype = DEFAULT_FILE_TYPE

    with picamera.PiCamera() as cam:
        cam.resolution = resolution
        cam.exposure_mode = exposure_mode

        cam.capture(stream, filetype)
        cam.close()

    return stream

# =========================================================
# Captures image and saves it to a file.
# Params:
# - filename        => string
# - resolution      => 2 integer tuple (x, y)
# - exposure_mode   => string
# Returns:          void
# =========================================================
def capture_to_file(
                    filename,
                    resolution = None,
                    exposure_mode = None,
                    ):
    if resolution is None:
        resolution = MAX_CAM_RESOLUTION
    if exposure_mode is None:
        exposure_mode = DEFAULT_EXPOSURE_MODE

    with picamera.PiCamera() as cam:
        cam.resolution = resolution
        cam.exposure_mode = exposure_mode

        cam.capture(filename)
        cam.close()

    return



#EOF
