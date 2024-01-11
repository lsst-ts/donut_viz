import batoid
import galsim
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize_scalar
from zernikePyramid import zernikePyramid
import yaml


def spotSize(optic, wavelength, nrad=5, naz=30, outer_field=1.75):
    thx, thy = batoid.utils.hexapolar(
        outer=np.deg2rad(outer_field),
        nrad=nrad,
        naz=naz,
    )

    # We'll use the 80th percentile of the RMS spot size over the field points
    mss = []
    for thx_, thy_ in zip(thx, thy):
        rays = batoid.RayVector.asPolar(
            optic, wavelength=wavelength,
            theta_x=thx_, theta_y=thy_,
            nrad=nrad*3, naz=naz*3,
        )
        rays = optic.trace(rays)
        xs = rays.x[~rays.vignetted]
        ys = rays.y[~rays.vignetted]
        xs -= np.mean(xs)
        ys -= np.mean(ys)
        mss.append(np.mean(np.square(xs) + np.square(ys)))

        return np.sqrt(np.quantile(mss, 0.8)) * 0.2/10e-6  # convert to arcsec


def spotSizeObjective(
    camera_z, optic, wavelength, nrad=5, naz=30, outer_field=1.75
):
    perturbed = optic.withGloballyShiftedOptic('LSSTCamera', [0, 0, camera_z])
    return spotSize(perturbed, wavelength, nrad, naz, outer_field)


def focus(optic, wavelength, nrad=5, naz=30, outer_field=1.75):
    return minimize_scalar(
        spotSizeObjective,
        bounds=(-1e-4, 1e-4),
        args=(optic, wavelength, nrad, naz, outer_field),
        options={'xatol':1e-8}
    )


def createIntrinsicZernikes():
    for f in "ugrizy":
        print("Filter:", f)
        # Use the effective wavelenght of the GalSim filters
        bandpass = galsim.Bandpass(f"LSST_{f}.dat", wave_type="nm")
        wavelength = bandpass.effective_wavelength * 1e-9  # convert to meters

        # telescope = batoid.Optic.fromYaml(f"Rubin_v3.12_{f}.yaml")
        telescope = batoid.Optic.fromYaml(f"LSST_{f}.yaml")
        focus_result = focus(telescope, wavelength)
        focus_z = focus_result.x
        focus_value = focus_result.fun
        telescope = telescope.withGloballyShiftedOptic('LSSTCamera', [0, 0, focus_z])
        print("Focus result:", focus_z, focus_value)

        thx, thy = batoid.utils.hexapolar(
            outer=1.82,
            nrad=15,
            naz=int(15*2*np.pi),
        )
        zk = np.empty((len(thx), 29))
        for izk, (thx_, thy_) in enumerate(zip(tqdm(thx), thy)):
            zk[izk] = batoid.zernike(
                telescope,
                np.deg2rad(thx_), np.deg2rad(thy_),
                wavelength,
                jmax=28,
                eps=0.612,
                nx=128,
            )*wavelength*1e6 # convert to microns

        basis = galsim.zernike.zernikeBasis(36, thx, thy, R_outer=1.82)
        # basis = galsim.zernike.zernikeBasis(55, thx, thy, R_outer=1.82)
        coefs, _, _, _ = np.linalg.lstsq(basis.T, zk, rcond=None)
        dzs = galsim.zernike.DoubleZernike(
            coefs,
            uv_outer=1.82, uv_inner=0.0,
            xy_outer=4.18, xy_inner=4.18*0.612
        )

        resid = np.array([z.coef for z in dzs(thx, thy)]) - zk
        print("Residuals quantiles (0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0):")
        print(np.quantile(resid, [0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0]))

        fig = zernikePyramid(
            thx, thy,
            zk.T[4:], cmap='seismic', s=2,
        )
        fig.savefig(f"zk_{f}.png")

        fig2 = zernikePyramid(
            thx, thy,
            resid.T[4:], cmap='seismic', s=2,
            vmin=-0.01, vmax=0.01
        )
        fig2.savefig(f"resid_{f}.png")
        print()

        with open(f"intrinsic_dz_{f}.yaml", 'w') as f:
            yaml.dump(coefs.tolist(), f)


if __name__ == "__main__":
    createIntrinsicZernikes()
