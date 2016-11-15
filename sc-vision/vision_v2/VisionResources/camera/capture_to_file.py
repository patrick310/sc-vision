import picamera # PiCamera()
import io #BytesIO()
import defaults

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
