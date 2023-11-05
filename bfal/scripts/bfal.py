import click
import bfal.config as cf

@click.group()
@click.option('--cam', '-c', type=int, default=cf.get(cf.CAM_TARGET), show_default=True, help='Target camera.')
@click.option('--res', '-r', type=click.Tuple((int, int)), default=(cf.get(cf.CAM_WIDTH), cf.get(cf.CAM_HEIGHT)), show_default=True, help='Camera Resolution')
@click.option('--override', '-o', is_flag=True, help='Save all overriden properties to config file.')
# @click.option('--crop', '-c', is_flag=True, default=(cf.get(cf.CAM_WIDTH), cf.get(cf.CAM_HEIGHT)), show_default=True, help='Camera Resolution')
def main(cam, res, override):
    """
        Built and Face recognition
    """
    cf.set(cf.CAM_TARGET, cam, override=override)
    cf.set(cf.CAM_WIDTH, res[0], override=override)
    cf.set(cf.CAM_HEIGHT, res[1], override=override)
    global save_config
    save_config = override

    if save_config:
        cf.save()


@main.command()
@click.option('--tol', '-t', type=int, default=cf.get(cf.CNFD_CALIB_TOL), show_default=True, help='Required number of steady value.')
@click.option('--dist', '-d', type=float, default=cf.get(cf.CNFD_VALUE), show_default=True, help='Value of real distance.')
@click.option('--unit', '-u', type=str, default=cf.get(cf.CNFD_UNIT), show_default=True, help='Unit of measurement used for real distance.')
def calibrate(tol, dist, unit):
    """
        Calibrate distance reference using two aruco card.
    """
    global save_config

    cf.set(cf.CNFD_CALIB_TOL, tol, override=save_config)
    cf.set(cf.CNFD_VALUE, dist, override=save_config)
    cf.set(cf.CNFD_UNIT, unit, override=save_config)

    if save_config:
        cf.save()

    from bfal.scripts import calibrate


@main.command()
@click.option('--port', '-p', type=str, default=cf.get(cf.SERIAL_PORT),help='serial connection port address')
@click.option('--baudrate', '-br', type=int, default=cf.get(cf.SERIAL_BAUDRATE), show_default=True,help='serial connection baudrate')
@click.option('--live-aref', '-lr', is_flag=True, default=False, help='use live aruco distance reference.')
@click.option('--faces-path', '-fp',type=click.Path(dir_okay=True, file_okay=False, exists=False),default=cf.get(cf.PATH_FACES), help="Directory of known faces.")
@click.option('--builts-path', '-bp',type=click.Path(dir_okay=False, file_okay=True, exists=False),default=cf.get(cf.PATH_BUILTS), help="File location of persons builts.")
# thresholds options
@click.option('--face-visibility', '-fv', type=float, default=cf.get(cf.TH_FACE_VISIBILITY), show_default=True, help="Minimum visibility required for each face point")
@click.option('--body-visibility', '-bv', type=float, default=cf.get(cf.TH_BODY_VISIBILITY), show_default=True, help="Minimum average visibility required for body points")
@click.option('--ankle-line', '-al', type=int, default=cf.get(cf.TH_ANKLE_LINE), show_default=True, help="Maximum tolerated distance between two ankle points based on their mid Y-axis")
@click.option('--shoulder-line', '-sl', type=float, default=cf.get(cf.TH_SHOULDER_LINE), show_default=True, help="Maximum tolerated distance between two shoulder points based on their mid Y-axis")
@click.option('--head-angle', '-hl', type=float, default=cf.get(cf.TH_HEAD_ANGLE), show_default=True, help="Allowed angle range for the head, measured from the nose to the mid-eye point")
@click.option('--knee-bend', '-kb', type=float, default=cf.get(cf.TH_KNEE_BEND), show_default=True, help="Tolerance for knee bend, calculated by comparing the distance of two endpoints to the sum of distances between each point")
@click.option('--built-tolerance', '-bt', type=int, default=cf.get(cf.TH_BUILT_TOLERANCE), show_default=True, help="Maximum tolerance for unit value based on the REFERENCE section")
@click.option('--aruco-line', '-arl', type=int, default=cf.get(cf.TH_ARUCO_LINE), show_default=True, help="Maximum tolerated distance between two Aruco midpoints based on their mid Y-axis")
@click.option('--aruco-body-line', '-abl', type=int, default=cf.get(cf.TH_ARUCO_BODY_LINE), show_default=True, help="Threshold for the maximum allowed distance between the bottom body point and the Aruco line reference")
@click.option('--serial-consistency', '-sc', type=int, default=cf.get(cf.TH_SERIAL_CONSISTENCY_REQ), show_default=True, help="Number of constant messages required before sending a serial message")
@click.option('--serial-window', '-sw', type=int, default=cf.get(cf.TH_SERIAL_WINDOW), show_default=True, help="Maximum time duration for a message to be considered valid as part of the constant message")
def detect(port, baudrate, live_aref, faces_path, builts_path,
        # thresholds
        face_visibility,
        body_visibility,
        ankle_line,
        shoulder_line,
        head_angle,
        knee_bend,
        built_tolerance,
        aruco_line,
        aruco_body_line,
        serial_consistency,
        serial_window,
    ):
    """
        Test built and face recognition.
    """
    global save_config

    cf.set(cf.SERIAL_PORT, port, override=save_config)
    cf.set(cf.SERIAL_BAUDRATE, baudrate, override=save_config)
    cf.set(cf.CNFD_USE_LIVE_REF, live_aref, override=save_config)
    cf.set(cf.PATH_FACES, faces_path, override=save_config)
    cf.set(cf.PATH_BUILTS, builts_path, override=save_config)

    cf.set(cf.TH_FACE_VISIBILITY, face_visibility, override=save_config)
    cf.set(cf.TH_BODY_VISIBILITY, body_visibility, override=save_config)
    cf.set(cf.TH_ANKLE_LINE, ankle_line, override=save_config)
    cf.set(cf.TH_SHOULDER_LINE, shoulder_line, override=save_config)
    cf.set(cf.TH_HEAD_ANGLE, head_angle, override=save_config)
    cf.set(cf.TH_KNEE_BEND, knee_bend, override=save_config)
    cf.set(cf.TH_BUILT_TOLERANCE, built_tolerance, override=save_config)
    cf.set(cf.TH_ARUCO_LINE, aruco_line, override=save_config)
    cf.set(cf.TH_ARUCO_BODY_LINE, aruco_body_line, override=save_config)
    cf.set(cf.TH_SERIAL_CONSISTENCY_REQ, serial_consistency, override=save_config)
    cf.set(cf.TH_SERIAL_WINDOW, serial_window, override=save_config)

    if save_config:
        cf.save()

    from bfal.scripts import detect