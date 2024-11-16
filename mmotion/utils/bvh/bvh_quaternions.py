import numpy as np


class Quaternions:
    """
    Quaternions is a wrapper around a numpy ndarray
    that allows it to act as if it were an narray of
    a quater data type.

    Therefore addition, subtraction, multiplication,
    division, negation, absolute, are all defined
    in terms of quater operations such as quater
    multiplication.

    This allows for much neater code and many routines
    which conceptually do the same thing to be written
    in the same way for point data and for rotation data.

    The Quaternions class has been desgined such that it
    should support broadcasting and slicing in all of the
    usual ways.
    """

    def __init__(self, qs):
        if isinstance(qs, np.ndarray):
            if len(qs.shape) == 1: qs = np.array([qs], dtype=np.float32)
            self.qs = qs
            return

        if isinstance(qs, Quaternions):
            self.qs = qs
            return

        raise TypeError('Quaternions must be constructed from iterable, numpy array, or Quaternions, not %s' % type(qs))

    def __str__(self):
        return "Quaternions(" + str(self.qs) + ")"

    def __repr__(self):
        return "Quaternions(" + repr(self.qs) + ")"

    """ Helper Methods for Broadcasting and Data extraction """

    @classmethod
    def _broadcast(cls, sqs, oqs, scalar=False):
        if isinstance(oqs, float): return sqs, oqs * np.ones(sqs.shape[:-1])

        ss = np.array(sqs.shape) if not scalar else np.array(sqs.shape[:-1])
        os = np.array(oqs.shape)

        if len(ss) != len(os):
            raise TypeError('Quaternions cannot broadcast together shapes %s and %s' % (sqs.shape, oqs.shape))

        if np.all(ss == os): return sqs, oqs

        if not np.all((ss == os) | (os == np.ones(len(os))) | (ss == np.ones(len(ss)))):
            raise TypeError('Quaternions cannot broadcast together shapes %s and %s' % (sqs.shape, oqs.shape))

        sqsn, oqsn = sqs.copy(), oqs.copy()

        for a in np.where(ss == 1)[0]: sqsn = sqsn.repeat(os[a], axis=a)
        for a in np.where(os == 1)[0]: oqsn = oqsn.repeat(ss[a], axis=a)

        return sqsn, oqsn

    """ Adding Quaterions is just Defined as Multiplication """

    def __add__(self, other):
        return self * other

    def __sub__(self, other):
        return self / other

    """ Quaterion Multiplication """

    def __mul__(self, other):
        """
        Quaternion multiplication has three main methods.

        When multiplying a Quaternions array by Quaternions
        normal quater multiplication is performed.

        When multiplying a Quaternions array by a vector
        array of the same shape, where the last axis is 3,
        it is assumed to be a Quaternion by 3D-Vector
        multiplication and the 3D-Vectors are rotated
        in space by the Quaternions.

        When multipplying a Quaternions array by a scalar
        or vector of different shape it is assumed to be
        a Quaternions by Scalars multiplication and the
        Quaternions are scaled using Slerp and the identity
        quaternions.
        """

        """ If Quaternions type do Quaternions * Quaternions """
        if isinstance(other, Quaternions):
            sqs, oqs = Quaternions._broadcast(self.qs, other.qs)

            q0 = sqs[..., 0];
            q1 = sqs[..., 1];
            q2 = sqs[..., 2];
            q3 = sqs[..., 3];
            r0 = oqs[..., 0];
            r1 = oqs[..., 1];
            r2 = oqs[..., 2];
            r3 = oqs[..., 3];

            qs = np.empty(sqs.shape)
            qs[..., 0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
            qs[..., 1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
            qs[..., 2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
            qs[..., 3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0

            return Quaternions(qs)

        """ If array type do Quaternions * Vectors """
        if isinstance(other, np.ndarray) and other.shape[-1] == 3:
            vs = Quaternions(np.concatenate([np.zeros(other.shape[:-1] + (1,)), other], axis=-1))

            return (self * (vs * -self)).imaginaries

        """ If float do Quaternions * Scalars """
        if isinstance(other, np.ndarray) or isinstance(other, float):
            return Quaternions.slerp(Quaternions.id_like(self), self, other)

        raise TypeError('Cannot multiply/add Quaternions with type %s' % str(type(other)))

    def __div__(self, other):
        """
        When a Quaternion type is supplied, division is defined
        as multiplication by the inverse of that Quaternion.

        When a scalar or vector is supplied it is defined
        as multiplicaion of one over the supplied value.
        Essentially a scaling.
        """

        if isinstance(other, Quaternions): return self * (-other)
        if isinstance(other, np.ndarray): return self * (1.0 / other)
        if isinstance(other, float): return self * (1.0 / other)
        raise TypeError('Cannot divide/subtract Quaternions with type %s' + str(type(other)))

    def __eq__(self, other):
        return self.qs == other.qs

    def __ne__(self, other):
        return self.qs != other.qs

    def __neg__(self):
        """ Invert Quaternions """
        return Quaternions(self.qs * np.array([[1, -1, -1, -1]]))

    def __abs__(self):
        """ Unify Quaternions To Single Pole """
        qabs = self.normalized().copy()
        top = np.sum((qabs.qs) * np.array([1, 0, 0, 0]), axis=-1)
        bot = np.sum((-qabs.qs) * np.array([1, 0, 0, 0]), axis=-1)
        qabs.qs[top < bot] = -qabs.qs[top < bot]
        return qabs

    def __iter__(self):
        return iter(self.qs)

    def __len__(self):
        return len(self.qs)

    def __getitem__(self, k):
        return Quaternions(self.qs[k])

    def __setitem__(self, k, v):
        self.qs[k] = v.qs

    @property
    def lengths(self):
        return np.sum(self.qs ** 2.0, axis=-1) ** 0.5

    @property
    def reals(self):
        return self.qs[..., 0]

    @property
    def imaginaries(self):
        return self.qs[..., 1:4]

    @property
    def shape(self):
        return self.qs.shape[:-1]

    def repeat(self, n, **kwargs):
        return Quaternions(self.qs.repeat(n, **kwargs))

    def normalized(self):
        return Quaternions(self.qs / self.lengths[..., np.newaxis])

    def log(self):
        norm = abs(self.normalized())
        imgs = norm.imaginaries
        lens = np.sqrt(np.sum(imgs ** 2, axis=-1))
        lens = np.arctan2(lens, norm.reals) / (lens + 1e-10)
        return imgs * lens[..., np.newaxis]

    def constrained(self, axis):

        rl = self.reals
        im = np.sum(axis * self.imaginaries, axis=-1)

        t1 = -2 * np.arctan2(rl, im) + np.pi
        t2 = -2 * np.arctan2(rl, im) - np.pi

        top = Quaternions.exp(axis[np.newaxis] * (t1[:, np.newaxis] / 2.0))
        bot = Quaternions.exp(axis[np.newaxis] * (t2[:, np.newaxis] / 2.0))
        img = self.dot(top) > self.dot(bot)

        ret = top.copy()
        ret[img] = top[img]
        ret[~img] = bot[~img]
        return ret

    def constrained_x(self):
        return self.constrained(np.array([1, 0, 0]))

    def constrained_y(self):
        return self.constrained(np.array([0, 1, 0]))

    def constrained_z(self):
        return self.constrained(np.array([0, 0, 1]))

    def dot(self, q):
        return np.sum(self.qs * q.qs, axis=-1)

    def copy(self):
        return Quaternions(np.copy(self.qs))

    def reshape(self, s):
        self.qs.reshape(s)
        return self

    def interpolate(self, ws):
        return Quaternions.exp(np.average(abs(self).log, axis=0, weights=ws))

    def euler(self, order='xyz'):  # fix the wrong convert, this should convert to world euler by default.

        q = self.normalized().qs
        q0 = q[..., 0]
        q1 = q[..., 1]
        q2 = q[..., 2]
        q3 = q[..., 3]
        es = np.zeros(self.shape + (3,))

        if order == 'xyz':
            es[..., 0] = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            es[..., 1] = np.arcsin((2 * (q0 * q2 - q3 * q1)).clip(-1, 1))
            es[..., 2] = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        elif order == 'yzx':
            es[..., 0] = np.arctan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
            es[..., 1] = np.arctan2(2 * (q2 * q0 - q1 * q3), q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0)
            es[..., 2] = np.arcsin((2 * (q1 * q2 + q3 * q0)).clip(-1, 1))
        else:
            raise NotImplementedError('Cannot convert from ordering %s' % order)

        """

        # These conversion don't appear to work correctly for Maya.
        # http://bediyap.com/programming/convert-quaternion-to-euler-rotations/

        if   order == 'xyz':
            es[fa + (0,)] = np.arctan2(2 * (q0 * q3 - q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q1 * q3 + q0 * q2)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
        elif order == 'yzx':
            es[fa + (0,)] = np.arctan2(2 * (q0 * q1 - q2 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q1 * q2 + q0 * q3)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q0 * q2 - q1 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
        elif order == 'zxy':
            es[fa + (0,)] = np.arctan2(2 * (q0 * q2 - q1 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q0 * q1 + q2 * q3)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q0 * q3 - q1 * q2), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) 
        elif order == 'xzy':
            es[fa + (0,)] = np.arctan2(2 * (q0 * q2 + q1 * q3), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q0 * q3 - q1 * q2)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3)
        elif order == 'yxz':
            es[fa + (0,)] = np.arctan2(2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q0 * q1 - q2 * q3)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q1 * q3 + q0 * q2), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
        elif order == 'zyx':
            es[fa + (0,)] = np.arctan2(2 * (q0 * q1 + q2 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
            es[fa + (1,)] = np.arcsin((2 * (q0 * q2 - q1 * q3)).clip(-1,1))
            es[fa + (2,)] = np.arctan2(2 * (q0 * q3 + q1 * q2), q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
        else:
            raise KeyError('Unknown ordering %s' % order)

        """

        # https://github.com/ehsan/ogre/blob/master/OgreMain/src/OgreMatrix3.cpp
        # Use this class and convert from matrix

        return es

    def average(self):

        if len(self.shape) == 1:

            import numpy.core.umath_tests as ut
            system = ut.matrix_multiply(self.qs[:, :, np.newaxis], self.qs[:, np.newaxis, :]).sum(axis=0)
            w, v = np.linalg.eigh(system)
            qiT_dot_qref = (self.qs[:, :, np.newaxis] * v[np.newaxis, :, :]).sum(axis=1)
            return Quaternions(v[:, np.argmin((1. - qiT_dot_qref ** 2).sum(axis=0))])

        else:

            raise NotImplementedError('Cannot average multi-dimensionsal Quaternions')

    def angle_axis(self):

        norm = self.normalized()
        s = np.sqrt(1 - (norm.reals ** 2.0))
        s[s == 0] = 0.001

        angles = 2.0 * np.arccos(norm.reals)
        axis = norm.imaginaries / s[..., np.newaxis]

        return angles, axis

    def transforms(self):

        qw = self.qs[..., 0]
        qx = self.qs[..., 1]
        qy = self.qs[..., 2]
        qz = self.qs[..., 3]

        x2 = qx + qx;
        y2 = qy + qy;
        z2 = qz + qz;
        xx = qx * x2;
        yy = qy * y2;
        wx = qw * x2;
        xy = qx * y2;
        yz = qy * z2;
        wy = qw * y2;
        xz = qx * z2;
        zz = qz * z2;
        wz = qw * z2;

        m = np.empty(self.shape + (3, 3))
        m[..., 0, 0] = 1.0 - (yy + zz)
        m[..., 0, 1] = xy - wz
        m[..., 0, 2] = xz + wy
        m[..., 1, 0] = xy + wz
        m[..., 1, 1] = 1.0 - (xx + zz)
        m[..., 1, 2] = yz - wx
        m[..., 2, 0] = xz - wy
        m[..., 2, 1] = yz + wx
        m[..., 2, 2] = 1.0 - (xx + yy)

        return m

    def ravel(self):
        return self.qs.ravel()

    @classmethod
    def id(cls, n):

        if isinstance(n, tuple):
            qs = np.zeros(n + (4,))
            qs[..., 0] = 1.0
            return Quaternions(qs)

        if isinstance(n, int):
            qs = np.zeros((n, 4))
            qs[:, 0] = 1.0
            return Quaternions(qs)

        raise TypeError('Cannot Construct Quaternion from %s type' % str(type(n)))

    @classmethod
    def id_like(cls, a):
        qs = np.zeros(a.shape + (4,))
        qs[..., 0] = 1.0
        return Quaternions(qs)

    @classmethod
    def exp(cls, ws):

        ts = np.sum(ws ** 2.0, axis=-1) ** 0.5
        ts[ts == 0] = 0.001
        ls = np.sin(ts) / ts

        qs = np.empty(ws.shape[:-1] + (4,))
        qs[..., 0] = np.cos(ts)
        qs[..., 1] = ws[..., 0] * ls
        qs[..., 2] = ws[..., 1] * ls
        qs[..., 3] = ws[..., 2] * ls

        return Quaternions(qs).normalized()

    @classmethod
    def slerp(cls, q0s, q1s, a):

        fst, snd = cls._broadcast(q0s.qs, q1s.qs)
        fst, a = cls._broadcast(fst, a, scalar=True)
        snd, a = cls._broadcast(snd, a, scalar=True)

        len = np.sum(fst * snd, axis=-1)

        neg = len < 0.0
        len[neg] = -len[neg]
        snd[neg] = -snd[neg]

        amount0 = np.zeros(a.shape)
        amount1 = np.zeros(a.shape)

        linear = (1.0 - len) < 0.01
        omegas = np.arccos(len[~linear])
        sinoms = np.sin(omegas)

        amount0[linear] = 1.0 - a[linear]
        amount1[linear] = a[linear]
        amount0[~linear] = np.sin((1.0 - a[~linear]) * omegas) / sinoms
        amount1[~linear] = np.sin(a[~linear] * omegas) / sinoms

        return Quaternions(
            amount0[..., np.newaxis] * fst +
            amount1[..., np.newaxis] * snd)

    @classmethod
    def between(cls, v0s, v1s):
        a = np.cross(v0s, v1s)
        w = np.sqrt((v0s ** 2).sum(axis=-1) * (v1s ** 2).sum(axis=-1)) + (v0s * v1s).sum(axis=-1)
        return Quaternions(np.concatenate([w[..., np.newaxis], a], axis=-1)).normalized()

    @classmethod
    def from_angle_axis(cls, angles, axis):
        axis = axis / (np.sqrt(np.sum(axis ** 2, axis=-1)) + 1e-10)[..., np.newaxis]
        sines = np.sin(angles / 2.0)[..., np.newaxis]
        cosines = np.cos(angles / 2.0)[..., np.newaxis]
        return Quaternions(np.concatenate([cosines, axis * sines], axis=-1))

    @classmethod
    def from_euler(cls, es, order='xyz', world=False):

        axis = {
            'x': np.array([1, 0, 0]),
            'y': np.array([0, 1, 0]),
            'z': np.array([0, 0, 1]),
        }

        q0s = Quaternions.from_angle_axis(es[..., 0], axis[order[0]])
        q1s = Quaternions.from_angle_axis(es[..., 1], axis[order[1]])
        q2s = Quaternions.from_angle_axis(es[..., 2], axis[order[2]])

        return (q2s * (q1s * q0s)) if world else (q0s * (q1s * q2s))

    @classmethod
    def from_transforms(cls, ts):

        d0, d1, d2 = ts[..., 0, 0], ts[..., 1, 1], ts[..., 2, 2]

        q0 = (d0 + d1 + d2 + 1.0) / 4.0
        q1 = (d0 - d1 - d2 + 1.0) / 4.0
        q2 = (-d0 + d1 - d2 + 1.0) / 4.0
        q3 = (-d0 - d1 + d2 + 1.0) / 4.0

        q0 = np.sqrt(q0.clip(0, None))
        q1 = np.sqrt(q1.clip(0, None))
        q2 = np.sqrt(q2.clip(0, None))
        q3 = np.sqrt(q3.clip(0, None))

        c0 = (q0 >= q1) & (q0 >= q2) & (q0 >= q3)
        c1 = (q1 >= q0) & (q1 >= q2) & (q1 >= q3)
        c2 = (q2 >= q0) & (q2 >= q1) & (q2 >= q3)
        c3 = (q3 >= q0) & (q3 >= q1) & (q3 >= q2)

        q1[c0] *= np.sign(ts[c0, 2, 1] - ts[c0, 1, 2])
        q2[c0] *= np.sign(ts[c0, 0, 2] - ts[c0, 2, 0])
        q3[c0] *= np.sign(ts[c0, 1, 0] - ts[c0, 0, 1])

        q0[c1] *= np.sign(ts[c1, 2, 1] - ts[c1, 1, 2])
        q2[c1] *= np.sign(ts[c1, 1, 0] + ts[c1, 0, 1])
        q3[c1] *= np.sign(ts[c1, 0, 2] + ts[c1, 2, 0])

        q0[c2] *= np.sign(ts[c2, 0, 2] - ts[c2, 2, 0])
        q1[c2] *= np.sign(ts[c2, 1, 0] + ts[c2, 0, 1])
        q3[c2] *= np.sign(ts[c2, 2, 1] + ts[c2, 1, 2])

        q0[c3] *= np.sign(ts[c3, 1, 0] - ts[c3, 0, 1])
        q1[c3] *= np.sign(ts[c3, 2, 0] + ts[c3, 0, 2])
        q2[c3] *= np.sign(ts[c3, 2, 1] + ts[c3, 1, 2])

        qs = np.empty(ts.shape[:-2] + (4,))
        qs[..., 0] = q0
        qs[..., 1] = q1
        qs[..., 2] = q2
        qs[..., 3] = q3

        return cls(qs)


# Calculate cross object of two 3D vectors.
def _fast_cross(a, b):
    return np.concatenate([
        a[..., 1:2] * b[..., 2:3] - a[..., 2:3] * b[..., 1:2],
        a[..., 2:3] * b[..., 0:1] - a[..., 0:1] * b[..., 2:3],
        a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1]], axis=-1)


# Make origin quaternions (No rotations)
def eye(shape, dtype=np.float32):
    return np.ones(list(shape) + [4], dtype=dtype) * np.asarray([1, 0, 0, 0], dtype=dtype)


# Return norm of quaternions
def length(x):
    return np.sqrt(np.sum(x * x, axis=-1))


# Make unit quaternions
def normalize(x, eps=1e-8):
    return x / (length(x)[..., None] + eps)


# Calculate inverse rotations
def inv(q):
    return np.array([1, -1, -1, -1], dtype=np.float32) * q


# Calculate the dot product of two quaternions
def dot(x, y):
    return np.sum(x * y, axis=-1)[..., None] if x.ndim > 1 else np.sum(x * y, axis=-1)


# Multiply two quaternions (return rotations).
def mul(x, y):
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    return np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)


def inv_mul(x, y):
    return mul(inv(x), y)


def mul_inv(x, y):
    return mul(x, inv(y))


# Multiply quaternions and vectors (return vectors).
def mul_vec(q, x):
    t = 2.0 * _fast_cross(q[..., 1:], x)
    return x + q[..., 0][..., None] * t + _fast_cross(q[..., 1:], t)


def inv_mul_vec(q, x):
    return mul_vec(inv(q), x)


def unroll(x):
    y = x.copy()
    for i in range(1, len(x)):
        d0 = np.sum(y[i] * y[i - 1], axis=-1)
        d1 = np.sum(-y[i] * y[i - 1], axis=-1)
        y[i][d0 < d1] = -y[i][d0 < d1]
    return y


# Calculate quaternions between two 3D vectors (x to y).
def between(x, y):
    return np.concatenate([
        np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))[..., None] +
        np.sum(x * y, axis=-1)[..., None],
        _fast_cross(x, y)], axis=-1)


def log(x, eps=1e-5):
    length = np.sqrt(np.sum(np.square(x[..., 1:]), axis=-1))[..., None]
    halfangle = np.where(length < eps, np.ones_like(length), np.arctan2(length, x[..., 0:1]) / length)
    return halfangle * x[..., 1:]


def exp(x, eps=1e-5):
    halfangle = np.sqrt(np.sum(np.square(x), axis=-1))[..., None]
    c = np.where(halfangle < eps, np.ones_like(halfangle), np.cos(halfangle))
    s = np.where(halfangle < eps, np.ones_like(halfangle), np.sinc(halfangle / np.pi))
    return np.concatenate([c, s * x], axis=-1)


# Calculate global space rotations and positions from local space.
def fk(lrot, lpos, parents):
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[..., i:i + 1, :]) + gp[parents[i]])
        gr.append(mul(gr[parents[i]], lrot[..., i:i + 1, :]))

    return np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)


def fk_rot(lrot, parents):
    gr = [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gr.append(mul(gr[parents[i]], lrot[..., i:i + 1, :]))

    return np.concatenate(gr, axis=-2)


# Calculate local space rotations and positions from global space.
def ik(grot, gpos, parents):
    return (
        np.concatenate([
            grot[..., :1, :],
            mul(inv(grot[..., parents[1:], :]), grot[..., 1:, :]),
        ], axis=-2),
        np.concatenate([
            gpos[..., :1, :],
            mul_vec(
                inv(grot[..., parents[1:], :]),
                gpos[..., 1:, :] - gpos[..., parents[1:], :]),
        ], axis=-2))


def ik_rot(grot, parents):
    return np.concatenate([grot[..., :1, :],
                           mul(inv(grot[..., parents[1:], :]), grot[..., 1:, :]),
                           ], axis=-2)


def fk_vel(lrot, lpos, lvel, lang, parents):
    gp, gr, gv, ga = [lpos[..., :1, :]], [lrot[..., :1, :]], [lvel[..., :1, :]], [lang[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(mul_vec(gr[parents[i]], lpos[..., i:i + 1, :]) + gp[parents[i]])
        gr.append(mul(gr[parents[i]], lrot[..., i:i + 1, :]))
        gv.append(mul_vec(gr[parents[i]], lvel[..., i:i + 1, :]) +
                  _fast_cross(ga[parents[i]], mul_vec(gr[parents[i]], lpos[..., i:i + 1, :])) +
                  gv[parents[i]])
        ga.append(mul_vec(gr[parents[i]], lang[..., i:i + 1, :]) + ga[parents[i]])

    return (
        np.concatenate(gr, axis=-2),
        np.concatenate(gp, axis=-2),
        np.concatenate(gv, axis=-2),
        np.concatenate(ga, axis=-2))


# Linear Interpolation of two vectors
def lerp(x, y, t):
    return (1 - t) * x + t * y


# LERP of quaternions
def quat_lerp(x, y, t):
    return normalize(lerp(x, y, t))


# Spherical linear interpolation of quaternions
def slerp(x, y, t):
    if t == 0:
        return x
    elif t == 1:
        return y

    if dot(x, y) < 0:
        y = - y
    ca = dot(x, y)
    theta = np.arccos(np.clip(ca, 0, 1))

    r = normalize(y - x * ca)

    return x * np.cos(theta * t) + r * np.sin(theta * t)


###################################################
# Calculate other rotations from other quaternions.
###################################################

# Calculate euler angles from quaternions.
def to_euler(x, order='zyx'):
    q0 = x[..., 0:1]
    q1 = x[..., 1:2]
    q2 = x[..., 2:3]
    q3 = x[..., 3:4]

    if order == 'zyx':

        return np.concatenate([
            np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)),
            np.arcsin((2 * (q0 * q2 - q3 * q1)).clip(-1, 1)),
            np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))], axis=-1)

    elif order == 'yzx':

        return np.concatenate([
            np.arctan2(2 * (q2 * q0 - q1 * q3), q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0),
            np.arcsin((2 * (q1 * q2 + q3 * q0)).clip(-1, 1)),
            np.arctan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)], axis=-1)

    elif order == 'zxy':

        return np.concatenate([
            np.arctan2(2 * (q0 * q3 - q1 * q2), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3),
            np.arcsin((2 * (q0 * q1 + q2 * q3)).clip(-1, 1)),
            np.arctan2(2 * (q0 * q2 - q1 * q3), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)], axis=-1)

    elif order == 'yxz':

        return np.concatenate([
            np.arctan2(2 * (q1 * q3 + q0 * q2), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3),
            np.arcsin((2 * (q0 * q1 - q2 * q3)).clip(-1, 1)),
            np.arctan2(2 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3)], axis=-1)

    else:
        raise NotImplementedError('Cannot convert from ordering %s' % order)


# Calculate rotation matrix from quaternions.
def to_xform(x):
    qw, qx, qy, qz = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]

    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2

    return np.concatenate([
        np.concatenate([1.0 - (yy + zz), xy - wz, xz + wy], axis=-1)[..., None, :],
        np.concatenate([xy + wz, 1.0 - (xx + zz), yz - wx], axis=-1)[..., None, :],
        np.concatenate([xz - wy, yz + wx, 1.0 - (xx + yy)], axis=-1)[..., None, :],
    ], axis=-2)


# Calculate 6d orthogonal rotation representation (ortho6d) from quaternions.
# https://github.com/papagina/RotationContinuity
def to_xform_xy(x):
    qw, qx, qy, qz = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]

    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2

    return np.concatenate([
        np.concatenate([1.0 - (yy + zz), xy - wz], axis=-1)[..., None, :],
        np.concatenate([xy + wz, 1.0 - (xx + zz)], axis=-1)[..., None, :],
        np.concatenate([xz - wy, yz + wx], axis=-1)[..., None, :],
    ], axis=-2)


# Calculate scaled angle axis from quaternions.
def to_scaled_angle_axis(x, eps=1e-5):
    return 2.0 * log(x, eps)


#############################################
# Calculate quaternions from other rotations.
#############################################

# Calculate quaternions from axis angles.
def from_angle_axis(angle, axis):
    c = np.cos(angle / 2.0)[..., None]
    s = np.sin(angle / 2.0)[..., None]
    q = np.concatenate([c, s * axis], axis=-1)
    return q


# Calculate quaternions from axis-angle.
def from_axis_angle(rots):
    angle = np.linalg.norm(rots, axis=-1)
    axis = rots / angle[..., None]
    return from_angle_axis(angle, axis)


# Calculate quaternions from euler angles.
def from_euler(e, order='zyx'):
    axis = {
        'x': np.asarray([1, 0, 0], dtype=np.float32),
        'y': np.asarray([0, 1, 0], dtype=np.float32),
        'z': np.asarray([0, 0, 1], dtype=np.float32)}

    q0 = from_angle_axis(e[..., 0], axis[order[0]])
    q1 = from_angle_axis(e[..., 1], axis[order[1]])
    q2 = from_angle_axis(e[..., 2], axis[order[2]])

    return mul(q0, mul(q1, q2))


# Calculate quaternions from rotation matrix.
def from_xform(ts):
    return normalize(
        np.where((ts[..., 2, 2] < 0.0)[..., None],
                 np.where((ts[..., 0, 0] > ts[..., 1, 1])[..., None],
                          np.concatenate([
                              (ts[..., 2, 1] - ts[..., 1, 2])[..., None],
                              (1.0 + ts[..., 0, 0] - ts[..., 1, 1] - ts[..., 2, 2])[..., None],
                              (ts[..., 1, 0] + ts[..., 0, 1])[..., None],
                              (ts[..., 0, 2] + ts[..., 2, 0])[..., None]], axis=-1),
                          np.concatenate([
                              (ts[..., 0, 2] - ts[..., 2, 0])[..., None],
                              (ts[..., 1, 0] + ts[..., 0, 1])[..., None],
                              (1.0 - ts[..., 0, 0] + ts[..., 1, 1] - ts[..., 2, 2])[..., None],
                              (ts[..., 2, 1] + ts[..., 1, 2])[..., None]], axis=-1)),
                 np.where((ts[..., 0, 0] < -ts[..., 1, 1])[..., None],
                          np.concatenate([
                              (ts[..., 1, 0] - ts[..., 0, 1])[..., None],
                              (ts[..., 0, 2] + ts[..., 2, 0])[..., None],
                              (ts[..., 2, 1] + ts[..., 1, 2])[..., None],
                              (1.0 - ts[..., 0, 0] - ts[..., 1, 1] + ts[..., 2, 2])[..., None]], axis=-1),
                          np.concatenate([
                              (1.0 + ts[..., 0, 0] + ts[..., 1, 1] + ts[..., 2, 2])[..., None],
                              (ts[..., 2, 1] - ts[..., 1, 2])[..., None],
                              (ts[..., 0, 2] - ts[..., 2, 0])[..., None],
                              (ts[..., 1, 0] - ts[..., 0, 1])[..., None]], axis=-1))))


# Calculate quaternions from ortho6d.
def from_xform_xy(x):
    c2 = _fast_cross(x[..., 0], x[..., 1])
    c2 = c2 / np.sqrt(np.sum(np.square(c2), axis=-1))[..., None]
    c1 = _fast_cross(c2, x[..., 0])
    c1 = c1 / np.sqrt(np.sum(np.square(c1), axis=-1))[..., None]
    c0 = x[..., 0]

    return from_xform(np.concatenate([
        c0[..., None],
        c1[..., None],
        c2[..., None]], axis=-1))


# Calculate quaternions from scaled angle axis.
def from_scaled_angle_axis(x, eps=1e-5):
    return exp(x / 2.0, eps)
