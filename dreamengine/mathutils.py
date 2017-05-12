import sys
import math
import numpy as np

from dreamengine import quaternion


def lerp_vec3(v1, v2, f):
    return (lerp(v1[0], v2[0], f),
            lerp(v1[1], v2[1], f),
            lerp(v1[2], v2[2], f))


def lerp(v1, v2, f):
    return v1 + ((v2 - v1) * f)


def matrix_multiply(mats):
    temp = mats[0]
    for m in mats[1:]:
        temp = temp @ m
    return temp


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


# jacked from some fuckin forum post from 2010

def transform(m, v):
    return np.asarray(m * np.asmatrix(v).T)[:, 0]


def magnitude(v):
    return math.sqrt(np.sum(v ** 2))


def normalize(v):
    m = magnitude(v)
    if m == 0:
        return v
    return v / m


def ortho(l, r, b, t, n, f):
    dx = r - l
    dy = t - b
    dz = f - n
    rx = -(r + l) / (r - l)
    ry = -(t + b) / (t - b)
    rz = -(f + n) / (f - n)
    return np.matrix([[2.0 / dx, 0, 0, rx],
                      [0, 2.0 / dy, 0, ry],
                      [0, 0, -2.0 / dz, rz],
                      [0, 0, 0, 1]],
                     'f')


def perspective(fovy, aspect, n, f):
    s = 1.0 / math.tan(fovy / 2.0)
    sx, sy = s / aspect, s
    zz = (f + n) / (n - f)
    zw = 2 * f * n / (n - f)
    return np.matrix([[sx, 0, 0, 0],
                      [0, sy, 0, 0],
                      [0, 0, zz, zw],
                      [0, 0, -1, 0]],
                     'f')


def frustum(x0, x1, y0, y1, z0, z1):
    a = (x1 + x0) / (x1 - x0)
    b = (y1 + y0) / (y1 - y0)
    c = -(z1 + z0) / (z1 - z0)
    d = -2 * z1 * z0 / (z1 - z0)
    sx = 2 * z0 / (x1 - x0)
    sy = 2 * z0 / (y1 - y0)
    return np.matrix([[sx, 0, a, 0],
                      [0, sy, b, 0],
                      [0, 0, c, d],
                      [0, 0, -1, 0]])


def translate(xyz):
    x, y, z = xyz
    return np.matrix([[1, 0, 0, x],
                      [0, 1, 0, y],
                      [0, 0, 1, z],
                      [0, 0, 0, 1]],
                     'f')


def scale(xyz):
    x, y, z = xyz
    return np.matrix([[x, 0, 0, 0],
                      [0, y, 0, 0],
                      [0, 0, z, 0],
                      [0, 0, 0, 1]],
                     'f')


def sincos(a):
    a = math.radians(a)
    return math.sin(a), math.cos(a)


def rotate(a, xyz):
    x, y, z = normalize(xyz)
    s, c = sincos(a)
    nc = 1 - c
    return np.matrix([[x * x * nc + c, x * y * nc - z * s, x * z * nc + y * s, 0],
                      [y * x * nc + z * s, y * y * nc + c, y * z * nc - x * s, 0],
                      [x * z * nc - y * s, y * z * nc + x * s, z * z * nc + c, 0],
                      [0, 0, 0, 1]],
                     'f')


def rotx(a):
    s, c = sincos(a)
    return np.matrix([[1, 0, 0, 0],
                      [0, c, -s, 0],
                      [0, s, c, 0],
                      [0, 0, 0, 1]])


def roty(a):
    s, c = sincos(a)
    return np.matrix([[c, 0, s, 0],
                      [0, 1, 0, 0],
                      [-s, 0, c, 0],
                      [0, 0, 0, 1]])


def rotz(a):
    s, c = sincos(a)
    return np.matrix([[c, -s, 0, 0],
                      [s, c, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])


def lookat(eye, target, up):
    F = target[:3] - eye[:3]
    f = normalize(F)
    U = normalize(up[:3])
    s = np.cross(f, U)
    u = np.cross(s, f)
    M = np.matrix(np.identity(4))
    M[:3, :3] = np.vstack([s, u, -f])
    T = translate(-eye)
    return M * T


def viewport(x, y, w, h):
    x, y, w, h = map(float, (x, y, w, h))
    return np.matrix([[w / 2, 0, 0, x + w / 2],
                      [0, h / 2, 0, y + h / 2],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])


def mat4_2_mat3(mat):
    # god i hate numpy
    mat = np.delete(mat, 3, 0)
    mat = np.delete(mat, 3, 1)
    return mat


def inverse(mat):
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        shape = mat.shape
        return np.zeros(shape, 'f')


def transpose(mat):
    return np.transpose(mat)


FRUSTRUM_THINGY = np.array((0, 0, 0, 1.0), 'f')


def frustrum_test(mvp):
    clip = mvp @ FRUSTRUM_THINGY
    x, y, z, w = clip[0, 0], clip[0, 1], clip[0, 2], clip[0, 3]
    return abs(x) < w and abs(y) < w and 0 < z < w


def decompose_matrix(mat):
    # I would call this in update_mvp and cache results, but it might be a bit slow
    t = mat[:3, 3]
    t = t.flatten().tolist()[0]
    # t = (t[0], t[2], -t[1])  # wtf is going on
    # t_mat = translate(t)

    # i think something here fucks up mat
    # trns = np.transpose(mat)
    # sx = vec3_length(trns[0, :-1].tolist()[0])
    # sy = vec3_length(trns[1, :-1].tolist()[0])
    # sz = vec3_length(trns[2, :-1].tolist()[0])
    #
    # s = (sx, sy, sz)
    # #s_mat = scale(s)
    # scl = np.transpose(np.matrix((s, s, s), 'f'))
    # r_mat = mat4_2_mat3(mat)
    # r_mat = safe_divide(r_mat, scl)
    # #r_mat = r_mat / scl
    # if not np.all(np.isfinite(r_mat)):
    #     r_mat = np.asmatrix(np.identity(3))
    # r = quaternion.quat_from_matrix(r_mat)
    # # return t_mat, s_mat, r_mat
    # return t, s, r
    return t, None, None


def safe_divide(m1, m2):
    # FUCK i hate numpy
    if np.all(np.isfinite(m2)) and np.count_nonzero(m2) == len(m2):
        return m1 / m2
    return np.identity(3)


def mat3_2_mat4(mat):
    mat = np.lib.pad(mat, (0, 1), 'constant')  # wow this was a hard find
    mat[3][3] = 1  # w component is identity
    return mat


def vec3_subtract(vec1, vec2):
    # lol
    return (vec1[0] - vec2[0], vec1[1] - vec2[1], vec1[2] - vec2[2])

def vec3_length(vec):
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
