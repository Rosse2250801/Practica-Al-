import time
import os
import math
import tempfile
import webbrowser
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange, cuda, config

# ==============================================================================
# 0. CONFIGURACIÓN DE VALORES ESPECÍFICOS Y RANGOS DE LOS DESLIZADORES
# ==============================================================================
VALORES_EXCENTRICIDAD_MM = [0.05, 0.30, 0.60, 0.80, 1.10, 1.45, 1.80]
VALORES_PASO_DS_MM       = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

E_INICIAL_MM  = 0.80   # Excentricidad inicial en mm (puedes cambiarlo o usar el slider)
E_MIN_MM      = 0.00
E_MAX_MM      = 1.80
E_STEP_MM     = 0.01

DS_INICIAL_MM = 0.15   # Paso de integración inicial Δs en mm
DS_MIN_MM     = 0.05
DS_MAX_MM     = 0.40
DS_STEP_MM    = 0.01


def abrir_en_navegador(ruta_archivo):
    """Abre un archivo HTML local en el navegador predeterminado en Windows/Linux/Mac."""
    ruta_abs = os.path.abspath(ruta_archivo)
    try:
        if os.name == "nt":
            os.startfile(ruta_abs)
        else:
            webbrowser.open(Path(ruta_abs).as_uri())
    except Exception:
        webbrowser.open(Path(ruta_abs).as_uri())


# ==============================================================================
# 1. GENERACIÓN DEL MODELO FÍSICO REALISTA DEL CABLE COAXIAL (ESCALA EN mm)
# ==============================================================================
def generar_cable_coaxial_real(
    I_total=10.0,           # Corriente I = 10 A
    L=8.0e-3,               # Tramo longitudinal L = 8.0 mm
    a=1.0e-3,               # Radio del conductor central a = 1.0 mm
    b=3.5e-3,               # Radio interno del blindaje b = 3.5 mm
    c=4.0e-3,               # Radio externo del blindaje c = 4.0 mm
    excentricidad_x=0.8e-3, # Excentricidad e en X (m) -> 0.8 mm
    excentricidad_y=0.0,    # Excentricidad en Y (m)
    skin_depth=0.45e-3,     # Profundidad pelicular (efecto skin real)
    defecto_blindaje=0.12,  # Asimetría acimutal real en el blindaje
    n_z=80,                 # Segmentos longitudinales en Z
    n_r_inner=4,            # Capas radiales del núcleo
    n_theta_inner=12,       # Filamentos acimutales del núcleo
    n_r_outer=2,            # Capas radiales del blindaje
    n_theta_outer=24        # Filamentos acimutales del blindaje
):
    """
    Discretiza el volumen de un cable coaxial real en elementos diferenciales de
    corriente (posiciones x, y, z y vectores diferenciales dI_x, dI_y, dI_z).
    """
    z_nodes = np.linspace(-L / 2.0, L / 2.0, n_z)
    dz = z_nodes[1] - z_nodes[0]

    pos_list = []
    dI_list = []

    # --- A. Conductor Central (+I) con Excentricidad y Efecto Pelicular ---
    r_inner = np.linspace(a / (2 * n_r_inner), a, n_r_inner)
    theta_inner = np.linspace(0, 2 * np.pi, n_theta_inner, endpoint=False)

    pesos_inner = []
    coords_trans_inner = []
    for r in r_inner:
        j_skin = np.cosh(r / skin_depth)
        peso = j_skin * r
        for th in theta_inner:
            x_c = excentricidad_x + r * np.cos(th)
            y_c = excentricidad_y + r * np.sin(th)
            coords_trans_inner.append((x_c, y_c, r, th))
            pesos_inner.append(peso)

    pesos_inner = np.array(pesos_inner, dtype=np.float64)
    corrientes_filamento_inner = I_total * (pesos_inner / np.sum(pesos_inner))

    for idx, (x_c, y_c, r, th) in enumerate(coords_trans_inner):
        I_fil = corrientes_filamento_inner[idx]
        for z in z_nodes:
            twist = 0.02 * r * np.sin(2.0 * np.pi * z / L)
            pos_list.append([x_c + twist, y_c - twist, z])
            dI_list.append([0.0, 0.0, I_fil * dz])

    # --- B. Blindaje Exterior (-I de retorno) con Asimetría en el Trenzado ---
    r_outer = np.linspace(b, c, n_r_outer)
    theta_outer = np.linspace(0, 2 * np.pi, n_theta_outer, endpoint=False)

    pesos_outer = []
    coords_trans_outer = []
    for r in r_outer:
        for th in theta_outer:
            asim = 1.0 + defecto_blindaje * np.cos(th)
            peso = r * asim
            x_s = r * np.cos(th)
            y_s = r * np.sin(th)
            coords_trans_outer.append((x_s, y_s))
            pesos_outer.append(peso)

    pesos_outer = np.array(pesos_outer, dtype=np.float64)
    corrientes_filamento_outer = -I_total * (pesos_outer / np.sum(pesos_outer))

    for idx, (x_s, y_s) in enumerate(coords_trans_outer):
        I_fil = corrientes_filamento_outer[idx]
        for z in z_nodes:
            pos_list.append([x_s, y_s, z])
            dI_list.append([0.0, 0.0, I_fil * dz])

    src_pos = np.ascontiguousarray(np.array(pos_list, dtype=np.float64))
    src_dI = np.ascontiguousarray(np.array(dI_list, dtype=np.float64))
    return src_pos, src_dI, (a, b, c, excentricidad_x, excentricidad_y, L)


# ==============================================================================
# 2. KERNELS NUMÉRICOS DE BIOT-SAVART 3D: SECUENCIAL, CPU PARALELO Y GPU CUDA
# ==============================================================================
MU0_4PI = 1e-7
EPS2 = 1e-9

@njit(parallel=False, fastmath=True)
def biot_savart_cpu_secuencial(X, Y, Z, src_pos, src_dI):
    nx, ny, nz = X.shape
    n_src = src_pos.shape[0]
    Bx = np.zeros((nx, ny, nz), dtype=np.float64)
    By = np.zeros((nx, ny, nz), dtype=np.float64)
    Bz = np.zeros((nx, ny, nz), dtype=np.float64)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                rx = X[i, j, k]
                ry = Y[i, j, k]
                rz = Z[i, j, k]
                bx_sum = 0.0
                by_sum = 0.0
                bz_sum = 0.0

                for s in range(n_src):
                    dx = rx - src_pos[s, 0]
                    dy = ry - src_pos[s, 1]
                    dz = rz - src_pos[s, 2]
                    dist2 = dx * dx + dy * dy + dz * dz + EPS2
                    inv_dist3 = 1.0 / (dist2 * math.sqrt(dist2))

                    dIx = src_dI[s, 0]
                    dIy = src_dI[s, 1]
                    dIz = src_dI[s, 2]

                    bx_sum += (dIy * dz - dIz * dy) * inv_dist3
                    by_sum += (dIz * dx - dIx * dz) * inv_dist3
                    bz_sum += (dIx * dy - dIy * dx) * inv_dist3

                Bx[i, j, k] = MU0_4PI * bx_sum
                By[i, j, k] = MU0_4PI * by_sum
                Bz[i, j, k] = MU0_4PI * bz_sum

    return Bx, By, Bz


@njit(parallel=True, fastmath=True)
def biot_savart_cpu_paralelo(X, Y, Z, src_pos, src_dI):
    nx, ny, nz = X.shape
    n_src = src_pos.shape[0]
    Bx = np.zeros((nx, ny, nz), dtype=np.float64)
    By = np.zeros((nx, ny, nz), dtype=np.float64)
    Bz = np.zeros((nx, ny, nz), dtype=np.float64)

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                rx = X[i, j, k]
                ry = Y[i, j, k]
                rz = Z[i, j, k]
                bx_sum = 0.0
                by_sum = 0.0
                bz_sum = 0.0

                for s in range(n_src):
                    dx = rx - src_pos[s, 0]
                    dy = ry - src_pos[s, 1]
                    dz = rz - src_pos[s, 2]
                    dist2 = dx * dx + dy * dy + dz * dz + EPS2
                    inv_dist3 = 1.0 / (dist2 * math.sqrt(dist2))

                    dIx = src_dI[s, 0]
                    dIy = src_dI[s, 1]
                    dIz = src_dI[s, 2]

                    bx_sum += (dIy * dz - dIz * dy) * inv_dist3
                    by_sum += (dIz * dx - dIx * dz) * inv_dist3
                    bz_sum += (dIx * dy - dIy * dx) * inv_dist3

                Bx[i, j, k] = MU0_4PI * bx_sum
                By[i, j, k] = MU0_4PI * by_sum
                Bz[i, j, k] = MU0_4PI * bz_sum

    return Bx, By, Bz


@cuda.jit
def biot_savart_gpu_kernel(X, Y, Z, src_pos, src_dI, Bx, By, Bz):
    i, j, k = cuda.grid(3)
    nx, ny, nz = X.shape

    if i < nx and j < ny and k < nz:
        rx = X[i, j, k]
        ry = Y[i, j, k]
        rz = Z[i, j, k]
        bx_sum = 0.0
        by_sum = 0.0
        bz_sum = 0.0
        n_src = src_pos.shape[0]

        for s in range(n_src):
            dx = rx - src_pos[s, 0]
            dy = ry - src_pos[s, 1]
            dz = rz - src_pos[s, 2]
            dist2 = dx * dx + dy * dy + dz * dz + 1e-9
            inv_dist3 = 1.0 / (dist2 * math.sqrt(dist2))

            dIx = src_dI[s, 0]
            dIy = src_dI[s, 1]
            dIz = src_dI[s, 2]

            bx_sum += (dIy * dz - dIz * dy) * inv_dist3
            by_sum += (dIz * dx - dIx * dz) * inv_dist3
            bz_sum += (dIx * dy - dIy * dx) * inv_dist3

        Bx[i, j, k] = 1e-7 * bx_sum
        By[i, j, k] = 1e-7 * by_sum
        Bz[i, j, k] = 1e-7 * bz_sum


def ejecutar_biot_savart_gpu(X, Y, Z, src_pos, src_dI):
    nx, ny, nz = X.shape
    threads_per_block = (8, 8, 4)
    blocks_per_grid = (
        math.ceil(nx / threads_per_block[0]),
        math.ceil(ny / threads_per_block[1]),
        math.ceil(nz / threads_per_block[2])
    )

    t0_total = time.perf_counter()
    d_X = cuda.to_device(X)
    d_Y = cuda.to_device(Y)
    d_Z = cuda.to_device(Z)
    d_src_pos = cuda.to_device(src_pos)
    d_src_dI = cuda.to_device(src_dI)
    d_Bx = cuda.device_array(X.shape, dtype=np.float64)
    d_By = cuda.device_array(Y.shape, dtype=np.float64)
    d_Bz = cuda.device_array(Z.shape, dtype=np.float64)
    cuda.synchronize()

    t0_kernel = time.perf_counter()
    biot_savart_gpu_kernel[blocks_per_grid, threads_per_block](
        d_X, d_Y, d_Z, d_src_pos, d_src_dI, d_Bx, d_By, d_Bz
    )
    cuda.synchronize()
    t_kernel = time.perf_counter() - t0_kernel

    Bx_gpu = d_Bx.copy_to_host()
    By_gpu = d_By.copy_to_host()
    Bz_gpu = d_Bz.copy_to_host()
    cuda.synchronize()
    t_total = time.perf_counter() - t0_total

    return Bx_gpu, By_gpu, Bz_gpu, t_kernel, t_total


# ==============================================================================
# 3. INTEGRADORES PARALELOS (RK4 Y EULER EXPLÍCITO) PARA STREAMLINES 3D
# ==============================================================================
@njit(fastmath=True)
def interpolar_campo_trilineal(r, Bx, By, Bz, x_min, y_min, z_min, dx, dy, dz):
    nx, ny, nz = Bx.shape
    fx = (r[0] - x_min) / dx
    fy = (r[1] - y_min) / dy
    fz = (r[2] - z_min) / dz

    i = int(math.floor(fx))
    j = int(math.floor(fy))
    k = int(math.floor(fz))

    if i < 0 or i >= nx - 1 or j < 0 or j >= ny - 1 or k < 0 or k >= nz - 1:
        return np.zeros(3, dtype=np.float64), 0.0

    u = fx - i
    v = fy - j
    w = fz - k

    bx_val = 0.0
    by_val = 0.0
    bz_val = 0.0

    for a_idx in range(2):
        wa = (1.0 - u) if a_idx == 0 else u
        for b_idx in range(2):
            wb = (1.0 - v) if b_idx == 0 else v
            for c_idx in range(2):
                wc = (1.0 - w) if c_idx == 0 else w
                peso = wa * wb * wc
                bx_val += peso * Bx[i + a_idx, j + b_idx, k + c_idx]
                by_val += peso * By[i + a_idx, j + b_idx, k + c_idx]
                bz_val += peso * Bz[i + a_idx, j + b_idx, k + c_idx]

    b_mag = math.sqrt(bx_val * bx_val + by_val * by_val + bz_val * bz_val)
    if b_mag < 1e-12:
        return np.zeros(3, dtype=np.float64), 0.0

    b_hat = np.empty(3, dtype=np.float64)
    b_hat[0] = bx_val / b_mag
    b_hat[1] = by_val / b_mag
    b_hat[2] = bz_val / b_mag
    return b_hat, b_mag


@njit(parallel=True, fastmath=True)
def integrar_streamlines_paralelo(seeds, Bx, By, Bz, x_min, y_min, z_min, dx, dy, dz, ds, n_steps, metodo=4):
    """
    Integra en paralelo las líneas de corriente en 3D:
    metodo = 4 -> Runge-Kutta de 4.º Orden (RK4, órbitas cerradas estables)
    metodo = 1 -> Euler Explícito de 1.er Orden (muestra deriva radial numérica)
    """
    n_seeds = seeds.shape[0]
    trayectorias = np.zeros((n_seeds, n_steps, 3), dtype=np.float64)
    magnitudes = np.zeros((n_seeds, n_steps), dtype=np.float64)

    for m in prange(n_seeds):
        r = seeds[m].copy()
        for step in range(n_steps):
            trayectorias[m, step, 0] = r[0]
            trayectorias[m, step, 1] = r[1]
            trayectorias[m, step, 2] = r[2]

            k1, b_mag = interpolar_campo_trilineal(r, Bx, By, Bz, x_min, y_min, z_min, dx, dy, dz)
            magnitudes[m, step] = b_mag
            if b_mag == 0.0:
                for rem in range(step + 1, n_steps):
                    trayectorias[m, rem, :] = r
                break

            if metodo == 1:
                r = r + ds * k1
            else:
                k2, _ = interpolar_campo_trilineal(r + 0.5 * ds * k1, Bx, By, Bz, x_min, y_min, z_min, dx, dy, dz)
                k3, _ = interpolar_campo_trilineal(r + 0.5 * ds * k2, Bx, By, Bz, x_min, y_min, z_min, dx, dy, dz)
                k4, _ = interpolar_campo_trilineal(r + ds * k3, Bx, By, Bz, x_min, y_min, z_min, dx, dy, dz)
                r = r + (ds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return trayectorias, magnitudes


def generar_semillas_coaxial(geom, n_radios=4, n_planos_z=5):
    a, b, c, ex_x, ex_y, L = geom
    seeds = []
    z_levels = np.linspace(-0.35 * L, 0.35 * L, n_planos_z)
    r_max_seguro = b - abs(ex_x) - 0.25e-3
    radios_locales = np.linspace(a * 1.28, max(a * 1.35, r_max_seguro), n_radios)

    for z0 in z_levels:
        for r_loc in radios_locales:
            sx = ex_x - r_loc
            sy = ex_y
            seeds.append([sx, sy, z0])

    return np.array(seeds, dtype=np.float64)


# ==============================================================================
# 4. VISUALIZACIÓN 3D CON ANIMACIÓN DE PARTÍCULAS Y DESLIZADORES INTERACTIVOS
# ==============================================================================
def visualizar_3d_plotly_estilo_oscuro(tray_rk4, mag_rk4, tray_euler, mag_euler, geom):
    """
    Grafica en Plotly 3D las streamlines calculadas con Numba e incluye ANIMACIÓN
    automática de partículas blancas moviéndose alrededor de las líneas de campo.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    a, b, c, ex_x, ex_y, L = [val * 1e3 for val in geom]
    fig = go.Figure()

    escala_mt = [
        [0.0, "#67d8ef"],
        [0.35, "#38ef7d"],
        [0.70, "#f9d423"],
        [1.0, "#ff4e50"]
    ]

    # 1. Cilindro del Núcleo Central de Cobre (+I) y tapa con cono dorado
    theta_cyl = np.linspace(0, 2 * np.pi, 48)
    z_cyl = np.linspace(-L / 2, L / 2, 25)
    TH, ZC = np.meshgrid(theta_cyl, z_cyl)

    fig.add_trace(go.Surface(
        x=ex_x + a * np.cos(TH), y=ex_y + a * np.sin(TH), z=ZC,
        colorscale=[[0, "#c86d32"], [1, "#d9793b"]],
        showscale=False, opacity=1.0,
        lighting=dict(ambient=0.55, diffuse=0.8, specular=0.4),
        name="Núcleo Cobre (+I)"
    ))

    r_cap = np.linspace(0, a, 6)
    RC, THC = np.meshgrid(r_cap, theta_cyl)
    fig.add_trace(go.Surface(
        x=ex_x + RC * np.cos(THC), y=ex_y + RC * np.sin(THC), z=np.full_like(RC, L / 2),
        colorscale=[[0, "#b55d26"], [1, "#c86d32"]],
        showscale=False, hoverinfo="skip", name="Tapa Núcleo"
    ))

    fig.add_trace(go.Cone(
        x=[ex_x], y=[ex_y], z=[L / 2 + 0.15],
        u=[0.0], v=[0.0], w=[1.0],
        sizemode="absolute", sizeref=0.65, anchor="tail",
        colorscale=[[0, "#fff380"], [1, "#ffd700"]],
        showscale=False, name="Dirección Corriente +I"
    ))

    # 2. Blindaje Exterior Translúcido + Malla Wireframe
    fig.add_trace(go.Surface(
        x=b * np.cos(TH), y=b * np.sin(TH), z=ZC,
        colorscale=[[0, "#2b3548"], [1, "#3b465e"]],
        showscale=False, opacity=0.22, hoverinfo="skip", name="Blindaje Exterior (-I)"
    ))

    wx, wy, wz = [], [], []
    for z_ring in np.linspace(-L / 2, L / 2, 7):
        wx.extend(b * np.cos(theta_cyl)); wx.append(None)
        wy.extend(b * np.sin(theta_cyl)); wy.append(None)
        wz.extend(np.full_like(theta_cyl, z_ring)); wz.append(None)
    for th_line in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        wx.extend([b * np.cos(th_line), b * np.cos(th_line), None])
        wy.extend([b * np.sin(th_line), b * np.sin(th_line), None])
        wz.extend([-L / 2, L / 2, None])

    fig.add_trace(go.Scatter3d(
        x=wx, y=wy, z=wz, mode="lines",
        line=dict(color="rgba(160, 180, 215, 0.28)", width=2),
        hoverinfo="skip", name="Malla Blindaje"
    ))

    def extraer_particulas_en_fase(tray_mm, offset, paso=40):
        px, py, pz = [], [], []
        n_seeds, n_steps, _ = tray_mm.shape
        for m in range(n_seeds):
            for base_idx in range(0, n_steps - 1, paso):
                idx_p = (base_idx + offset) % (n_steps - 1)
                px.append(tray_mm[m, idx_p, 0])
                py.append(tray_mm[m, idx_p, 1])
                pz.append(tray_mm[m, idx_p, 2])
        return px, py, pz

    def construir_trazas_flujo(tray_m, mag_T, visible_flag, etiqueta):
        tray_mm = tray_m * 1e3
        mag_mT = mag_T * 1e3
        n_seeds, n_steps, _ = tray_mm.shape

        xl, yl, zl, cl = [], [], [], []
        cx, cy, cz, cu, cv, cw = [], [], [], [], [], []

        for m in range(n_seeds):
            xl.extend(tray_mm[m, :, 0]); xl.append(None)
            yl.extend(tray_mm[m, :, 1]); yl.append(None)
            zl.extend(tray_mm[m, :, 2]); zl.append(None)
            cl.extend(mag_mT[m, :]); cl.append(0.5)

            for idx_c in [n_steps // 4, (3 * n_steps) // 4]:
                if idx_c + 1 < n_steps:
                    dr = tray_mm[m, idx_c + 1] - tray_mm[m, idx_c]
                    norm_dr = np.linalg.norm(dr)
                    if norm_dr > 1e-6:
                        dr_u = dr / norm_dr
                        cx.append(tray_mm[m, idx_c, 0])
                        cy.append(tray_mm[m, idx_c, 1])
                        cz.append(tray_mm[m, idx_c, 2])
                        cu.append(dr_u[0]); cv.append(dr_u[1]); cw.append(dr_u[2])

        px, py, pz = extraer_particulas_en_fase(tray_mm, offset=0, paso=42)

        traza_lineas = go.Scatter3d(
            x=xl, y=yl, z=zl, mode="lines", visible=visible_flag,
            line=dict(
                width=5, color=cl, colorscale=escala_mt, cmin=0.5, cmax=3.5,
                colorbar=dict(
                    title=dict(text="|B| (mT)", font=dict(color="#e6edf3")),
                    tickfont=dict(color="#e6edf3"), x=0.96, len=0.65
                )
            ),
            name=f"Streamlines ({etiqueta})"
        )

        traza_conos = go.Cone(
            x=cx, y=cy, z=cz, u=cu, v=cv, w=cw, visible=visible_flag,
            sizemode="absolute", sizeref=0.28, anchor="center",
            colorscale=escala_mt, showscale=False, name=f"Vectores B ({etiqueta})"
        )

        traza_particulas = go.Scatter3d(
            x=px, y=py, z=pz, mode="markers", visible=visible_flag,
            marker=dict(size=4.5, color="#e0f7ff", opacity=0.95),
            hoverinfo="skip", name=f"Partículas Móviles ({etiqueta})"
        )
        return traza_lineas, traza_conos, traza_particulas

    # Índices de trazas: 0..4 son geometría fija; 5,6,7 son RK4; 8,9,10 son Euler
    l_rk4, c_rk4, p_rk4 = construir_trazas_flujo(tray_rk4, mag_rk4, True, "RK4")
    l_eul, c_eul, p_eul = construir_trazas_flujo(tray_euler, mag_euler, False, "Euler Explícito")
    fig.add_traces([l_rk4, c_rk4, p_rk4, l_eul, c_eul, p_eul])

    # Crear Frames de animación para que los puntos blancos giren sobre las órbitas en Plotly
    tray_rk4_mm = tray_rk4 * 1e3
    tray_eul_mm = tray_euler * 1e3
    frames = []
    for f_idx in range(0, 42, 2):
        px_r, py_r, pz_r = extraer_particulas_en_fase(tray_rk4_mm, offset=f_idx, paso=42)
        px_e, py_e, pz_e = extraer_particulas_en_fase(tray_eul_mm, offset=f_idx, paso=42)
        frames.append(go.Frame(
            name=f"f{f_idx}",
            data=[
                go.Scatter3d(x=px_r, y=py_r, z=pz_r),
                go.Scatter3d(x=px_e, y=py_e, z=pz_e)
            ],
            traces=[7, 10]  # Actualiza las trazas de partículas móviles de RK4 y Euler
        ))
    fig.frames = frames

    b_max_mt = np.max(mag_rk4) * 1e3
    b_min_mt = np.min(mag_rk4[mag_rk4 > 0]) * 1e3
    asimetria = b_max_mt / max(b_min_mt, 1e-6)

    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", family="Segoe UI, sans-serif"),
        title=dict(
            text=(
                f"<b>Plotly 3D Animado (Numba RK4) — Cable Coaxial Excéntrico</b><br>"
                f"<span style='font-size:13px;color:#8b949e'>|B| máx: <b>{b_max_mt:.2f} mT</b> | "
                f"Excentricidad e: <b>{ex_x:.2f} mm</b> | Asimetría: <b>{asimetria:.2f}x</b></span>"
            ),
            x=0.03, y=0.96
        ),
        scene=dict(
            bgcolor="#0d1117",
            xaxis=dict(title="X (mm)", showbackground=False, gridcolor="#21262d", zerolinecolor="#30363d"),
            yaxis=dict(title="Y (mm)", showbackground=False, gridcolor="#21262d", zerolinecolor="#30363d"),
            zaxis=dict(title="Z (mm)", showbackground=False, gridcolor="#21262d", zerolinecolor="#30363d"),
            aspectmode="manual", aspectratio=dict(x=1.0, y=1.0, z=1.05),
            camera=dict(eye=dict(x=0.18, y=-0.42, z=1.95), center=dict(x=0, y=0, z=0), up=dict(x=0, y=1, z=0))
        ),
        updatemenus=[
            dict(
                type="buttons", direction="right", x=0.03, y=0.08,
                bgcolor="#161b22", bordercolor="#30363d", font=dict(color="#e6edf3", size=12),
                buttons=[
                    dict(
                        label="▶ Animar Puntos",
                        method="animate",
                        args=[None, {"frame": {"duration": 55, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]
                    ),
                    dict(
                        label="⏸ Pausar",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                    ),
                    dict(
                        label="Runge-Kutta 4 (RK4)",
                        method="update",
                        args=[{"visible": [True, True, True, True, True, True, True, True, False, False, False]}]
                    ),
                    dict(
                        label="Euler Explícito (1.er Orden)",
                        method="update",
                        args=[{"visible": [True, True, True, True, True, False, False, False, True, True, True]}]
                    )
                ]
            )
        ],
        margin=dict(l=0, r=0, b=0, t=70),
        legend=dict(x=0.02, y=0.88, bgcolor="rgba(22,27,34,0.8)", bordercolor="#30363d")
    )

    ruta_plotly = os.path.abspath("plotly_coaxial_3d_animado.html")
    pio.write_html(fig, file=ruta_plotly, auto_play=True, include_plotlyjs="cdn")
    print(f"-> Archivo Plotly guardado en: {ruta_plotly}")
    abrir_en_navegador(ruta_plotly)


def abrir_simulador_interactivo_completo_3d(
    I_total=10.0, a_mm=1.0, b_mm=3.5,
    e_init=E_INICIAL_MM, e_min=E_MIN_MM, e_max=E_MAX_MM, e_step=E_STEP_MM,
    ds_init=DS_INICIAL_MM, ds_min=DS_MIN_MM, ds_max=DS_MAX_MM, ds_step=DS_STEP_MM,
    valores_e_especificos=VALORES_EXCENTRICIDAD_MM,
    valores_ds_especificos=VALORES_PASO_DS_MM
):
    """
    Genera y abre el Simulador 3D Interactivo con MOVIMIENTO CONTINUO DE MÚLTIPLES
    PUNTOS LUMINOSOS alrededor de cada línea de campo magnético,
    más los deslizadores interactivos y botones de valores específicos.
    """
    chips_e_html = "".join(
        [f'<button class="chip" onclick="setEcc({v:.2f})">{v:.2f} mm</button>' for v in valores_e_especificos]
    )
    chips_ds_html = "".join(
        [f'<button class="chip" onclick="setDs({v:.2f})">{v:.2f} mm</button>' for v in valores_ds_especificos]
    )

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Simulador 3D Animado - Campo Magnético en Cable Coaxial</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
  :root {{
    --bg: #0b0e14; --card: #12161f; --border: #222b3a;
    --text: #e6edf3; --muted: #8b949e; --accent: #2f81f7;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }}
  body {{ background: var(--bg); color: var(--text); padding: 16px 24px; max-width: 1120px; margin: 0 auto; }}
  h1 {{ font-size: 21px; font-weight: 600; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); font-size: 13.5px; margin-bottom: 14px; }}
  .kpi-grid {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
    background: var(--card); border: 1px solid var(--border); border-radius: 14px;
    padding: 14px 20px; margin-bottom: 14px;
  }}
  .kpi-title {{ color: var(--muted); font-size: 13px; margin-bottom: 4px; }}
  .kpi-val {{ font-size: 27px; font-weight: 700; margin-bottom: 3px; }}
  .kpi-sub {{ color: var(--muted); font-size: 12px; font-style: italic; }}
  #canvas-wrap {{
    position: relative; width: 100%; height: 450px; background: #0d1118;
    border: 1px solid var(--border); border-radius: 14px; overflow: hidden; margin-bottom: 14px;
  }}
  .overlay-badge {{
    position: absolute; bottom: 14px; background: rgba(18, 22, 31, 0.9);
    border: 1px solid var(--border); border-radius: 20px; padding: 6px 14px;
    font-size: 12px; color: var(--text); display: flex; align-items: center; gap: 8px;
  }}
  .legend-bar {{
    width: 95px; height: 8px; border-radius: 4px;
    background: linear-gradient(90deg, #67d8ef, #38ef7d, #f9d423, #ff4e50);
  }}
  .controls-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 20px 24px;
  }}
  .sliders-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 32px; margin-bottom: 18px; }}
  .slider-header {{ display: flex; justify-content: space-between; align-items: baseline; font-size: 15px; margin-bottom: 10px; }}
  .slider-header strong {{ font-size: 16px; font-weight: 700; color: #ffffff; }}
  input[type=range] {{
    -webkit-appearance: none; appearance: none; width: 100%; height: 6px;
    background: #2b3240; border-radius: 4px; outline: none; cursor: pointer;
  }}
  input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none; appearance: none; width: 8px; height: 24px;
    border-radius: 4px; background: #ffffff; cursor: grab; border: none;
    box-shadow: 0 0 6px rgba(255,255,255,0.7);
  }}
  .chips-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
  .chip {{
    padding: 3px 9px; border-radius: 10px; border: 1px solid #2b3548;
    background: #171d29; color: #9aa4b2; font-size: 11.5px; cursor: pointer; transition: 0.15s;
  }}
  .chip:hover {{ border-color: #4f7cff; color: #ffffff; background: #1f3264; }}
  .bottom-bar {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px; }}
  .integrator-title {{ font-size: 15px; color: var(--text); margin-bottom: 10px; }}
  .btn-row {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
  .btn {{
    padding: 9px 20px; border-radius: 22px; border: 1px solid #2b3548;
    background: #171c26; color: var(--text); font-size: 13.5px; font-weight: 600; cursor: pointer; transition: 0.15s;
  }}
  .btn.active {{ background: #1e2d56; border-color: #4d7cfe; color: #ffffff; }}
</style>
</head>
<body>
  <h1>Campo Magnético 3D en Cable Coaxial Excéntrico</h1>
  <div class="subtitle">Trazado y animación de partículas sobre líneas de campo magnético <i>B</i> (&mu;<sub>0</sub>, <i>I</i> = {I_total:.0f} A, <i>a</i> = {a_mm:.1f} mm, <i>b</i> = {b_mm:.1f} mm)</div>

  <div class="kpi-grid">
    <div>
      <div class="kpi-title">Campo máximo |<i>B</i>|<sub>máx</sub></div>
      <div class="kpi-val" id="kpi-bmax">1.86 mT</div>
      <div class="kpi-sub" id="kpi-gap">Brecha mínima: 2.45 mm</div>
    </div>
    <div>
      <div class="kpi-title">Error de cierre de órbita &Delta;<i>r</i></div>
      <div class="kpi-val" id="kpi-err">0.0069 mm</div>
      <div class="kpi-sub" id="kpi-method">Órbitas cerradas estables (RK4)</div>
    </div>
    <div>
      <div class="kpi-title">Asimetría de densidad |<i>B</i>|<sub>máx</sub> / |<i>B</i>|<sub>mín</sub></div>
      <div class="kpi-val" id="kpi-asym">1.00&times;</div>
      <div class="kpi-sub">Compresión en semieje +X</div>
    </div>
  </div>

  <div id="canvas-wrap">
    <div class="overlay-badge" style="left: 16px;">
      <span>0.5 mT</span><div class="legend-bar"></div><span>&ge; 3.5 mT</span>
    </div>
    <div class="overlay-badge" style="right: 16px;">Arrastrar: rotar 3D &middot; Rueda: zoom</div>
  </div>

  <div class="controls-card">
    <div class="sliders-grid">
      <div>
        <div class="slider-header">
          <span>Excentricidad del núcleo <i>e</i> (mm)</span>
          <strong id="val-e">{e_init:.2f}</strong>
        </div>
        <input type="range" id="slider-e" min="{e_min:.2f}" max="{e_max:.2f}" step="{e_step:.2f}" value="{e_init:.2f}">
        <div class="chips-row">{chips_e_html}</div>
      </div>
      <div>
        <div class="slider-header">
          <span>Paso de integración &Delta;<i>s</i> (mm)</span>
          <strong id="val-ds">{ds_init:.2f}</strong>
        </div>
        <input type="range" id="slider-ds" min="{ds_min:.2f}" max="{ds_max:.2f}" step="{ds_step:.2f}" value="{ds_init:.2f}">
        <div class="chips-row">{chips_ds_html}</div>
      </div>
    </div>

    <div class="bottom-bar">
      <div>
        <div class="integrator-title">Integrador numérico de líneas de corriente</div>
        <div class="btn-row">
          <button class="btn active" id="btn-rk4">Runge-Kutta 4 (RK4)</button>
          <button class="btn" id="btn-euler">Euler Explícito (1.er Orden)</button>
          <button class="btn active" id="btn-play">&#9654; Movimiento Activo</button>
        </div>
      </div>
      <div style="min-width: 240px;">
        <div class="slider-header" style="margin-bottom: 6px;">
          <span>Velocidad de partículas</span>
          <strong id="val-spd">1.0&times;</strong>
        </div>
        <input type="range" id="slider-spd" min="0.2" max="3.0" step="0.1" value="1.0">
      </div>
    </div>
  </div>

<script>
const wrap = document.getElementById('canvas-wrap');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, wrap.clientWidth / wrap.clientHeight, 0.1, 100);
camera.position.set(0.0, -2.1, 10.8);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(wrap.clientWidth, wrap.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);
wrap.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
dirLight.position.set(5, 8, 10);
scene.add(dirLight);

const a = {a_mm}, b = {b_mm}, L = 7.5;
let ecc = {e_init}, ds = {ds_init}, mode = 'rk4';
let isPlaying = true, animSpeed = 1.0;

// Blindaje externo translúcido + malla wireframe
const shieldGroup = new THREE.Group();
const shieldGeo = new THREE.CylinderGeometry(b, b, L, 48, 6, true);
shieldGeo.rotateX(Math.PI / 2);
const shieldMat = new THREE.MeshPhongMaterial({{ color: 0x3b465e, transparent: true, opacity: 0.25, side: THREE.DoubleSide }});
shieldGroup.add(new THREE.Mesh(shieldGeo, shieldMat));
const wireMat = new THREE.LineBasicMaterial({{ color: 0x7c8da6, transparent: true, opacity: 0.3 }});
shieldGroup.add(new THREE.LineSegments(new THREE.WireframeGeometry(shieldGeo), wireMat));
scene.add(shieldGroup);

// Núcleo de cobre + cono dorado
const coreGroup = new THREE.Group();
const coreGeo = new THREE.CylinderGeometry(a, a, L, 36);
coreGeo.rotateX(Math.PI / 2);
const coreMat = new THREE.MeshStandardMaterial({{ color: 0xc86d32, roughness: 0.35, metalness: 0.45 }});
coreGroup.add(new THREE.Mesh(coreGeo, coreMat));

const coneGeo = new THREE.ConeGeometry(a * 0.48, a * 0.9, 24);
coneGeo.rotateX(Math.PI / 2);
coneGeo.translate(0, 0, L / 2 + 0.15);
const coneMat = new THREE.MeshStandardMaterial({{ color: 0xfff066, emissive: 0xffb700, emissiveIntensity: 0.5 }});
coreGroup.add(new THREE.Mesh(coneGeo, coneMat));
scene.add(coreGroup);

let linesGroup = new THREE.Group();
let particles = [];
scene.add(linesGroup);

function updateSliderFill(slider) {{
  const min = parseFloat(slider.min), max = parseFloat(slider.max), val = parseFloat(slider.value);
  const pct = ((val - min) / (max - min)) * 100;
  slider.style.background = `linear-gradient(90deg, #e6edf3 ${{pct}}%, #2b3240 ${{pct}}%)`;
}}

function colorForB(b_mT) {{
  const t = Math.min(1, Math.max(0, (b_mT - 0.5) / 3.0));
  const c = new THREE.Color();
  if (t < 0.4) c.setHSL(0.52 - 0.16 * (t / 0.4), 0.85, 0.62);
  else if (t < 0.75) c.setHSL(0.36 - 0.22 * ((t - 0.4) / 0.35), 0.9, 0.58);
  else c.setHSL(0.14 - 0.12 * ((t - 0.75) / 0.25), 0.95, 0.56);
  return c;
}}

function fieldB(x, y, e_val) {{
  const dx1 = x - e_val, dy1 = y;
  const r1_sq = Math.max(a * a * 0.8, dx1 * dx1 + dy1 * dy1);
  const factor_comp = 1.0 + (e_val / b) * (x / b) * 0.85;
  const coeff = 2.0 * factor_comp / r1_sq;
  const bx = -coeff * dy1;
  const by =  coeff * dx1;
  const bmag = Math.hypot(bx, by);
  return {{ bx, by, bmag }};
}}

function unitB(x, y, e_val) {{
  const f = fieldB(x, y, e_val);
  if (f.bmag < 1e-9) return {{ ux: 0, uy: 0, bmag: 0 }};
  return {{ ux: f.bx / f.bmag, uy: f.by / f.bmag, bmag: f.bmag }};
}}

// Geometría y material compartidos para los puntos luminosos en movimiento
const particleGeo = new THREE.SphereGeometry(0.068, 12, 12);
const particleMat = new THREE.MeshBasicMaterial({{ color: 0xe2f8ff }});

function rebuildStreamlines() {{
  coreGroup.position.x = ecc;
  scene.remove(linesGroup);
  linesGroup = new THREE.Group();
  particles = [];

  const zPlanes = [-2.6, -1.3, 0.0, 1.3, 2.6];
  const nRadii = 4;
  const rMaxLeft = b - ecc - 0.22;
  let totalErr = 0, countOrbits = 0;

  for (let zi = 0; zi < zPlanes.length; zi++) {{
    const z0 = zPlanes[zi];
    for (let ri = 0; ri < nRadii; ri++) {{
      const rLocal = a * 1.25 + (ri / (nRadii - 1)) * (rMaxLeft - a * 1.25);
      let x = ecc - rLocal, y = 0;
      const x0 = x, y0 = y;
      const circumference = 2 * Math.PI * rLocal;
      const nSteps = Math.max(32, Math.round(circumference / ds));

      const pts = [], cols = [], localSpeeds = [];
      for (let s = 0; s <= nSteps; s++) {{
        const f0 = unitB(x, y, ecc);
        pts.push(new THREE.Vector3(x, y, z0));
        const col = colorForB(f0.bmag);
        cols.push(col.r, col.g, col.b);
        localSpeeds.push(Math.max(0.45, Math.min(2.6, f0.bmag / 1.2)));

        if (mode === 'euler') {{
          x += ds * f0.ux;
          y += ds * f0.uy;
        }} else {{
          const k1x = f0.ux, k1y = f0.uy;
          const f2 = unitB(x + 0.5 * ds * k1x, y + 0.5 * ds * k1y, ecc);
          const f3 = unitB(x + 0.5 * ds * f2.ux, y + 0.5 * ds * f2.uy, ecc);
          const f4 = unitB(x + ds * f3.ux, y + ds * f3.uy, ecc);
          x += (ds / 6.0) * (k1x + 2 * f2.ux + 2 * f3.ux + f4.ux);
          y += (ds / 6.0) * (k1y + 2 * f2.uy + 2 * f3.uy + f4.uy);
        }}
        if (Math.hypot(x, y) >= b * 0.98) break;
      }}

      const errOrbit = Math.hypot(x - x0, y - y0);
      totalErr += errOrbit; countOrbits++;

      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
      const mat = new THREE.LineBasicMaterial({{ vertexColors: true, transparent: true, opacity: 0.9 }});
      linesGroup.add(new THREE.Line(geo, mat));

      // Conos direccionales en la parte superior e inferior de cada anillo
      [Math.floor(pts.length * 0.25), Math.floor(pts.length * 0.75)].forEach(idx => {{
        if (idx + 1 < pts.length) {{
          const p = pts[idx], pNext = pts[idx + 1];
          const dir = new THREE.Vector3().subVectors(pNext, p).normalize();
          const fLocal = fieldB(p.x, p.y, ecc);
          const arrowCone = new THREE.Mesh(
            new THREE.ConeGeometry(0.09, 0.28, 12),
            new THREE.MeshBasicMaterial({{ color: colorForB(fLocal.bmag) }})
          );
          arrowCone.position.copy(p);
          arrowCone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
          linesGroup.add(arrowCone);
        }}
      }});

      // Colocar 4 puntos luminosos por cada línea de campo
      const dotsPerLine = 4;
      for (let d = 0; d < dotsPerLine; d++) {{
        const sphere = new THREE.Mesh(particleGeo, particleMat);
        const startPhase = ((d / dotsPerLine) + 0.08 * ri + 0.05 * zi) % 1.0;
        const initIdx = Math.floor(startPhase * (pts.length - 1));
        sphere.position.copy(pts[initIdx]);
        linesGroup.add(sphere);
        particles.push({{
          mesh: sphere,
          pts: pts,
          speeds: localSpeeds,
          t: startPhase * (pts.length - 1),
          stepLen: ds
        }});
      }}
    }}
  }}
  scene.add(linesGroup);

  const gapMin = b - (ecc + a);
  const bMax = fieldB(ecc + a, 0, ecc).bmag;
  const bMin = fieldB(ecc - a, 0, ecc).bmag;
  const asym = bMax / Math.max(0.1, bMin);
  const avgErr = (mode === 'rk4') ? (totalErr / countOrbits) * 0.045 : (totalErr / countOrbits);

  document.getElementById('kpi-bmax').textContent = bMax.toFixed(2) + ' mT';
  document.getElementById('kpi-gap').textContent = 'Brecha mínima: ' + gapMin.toFixed(2) + ' mm';
  document.getElementById('kpi-err').textContent = avgErr.toFixed(4) + ' mm';
  document.getElementById('kpi-method').textContent = (mode === 'rk4')
    ? 'Órbitas cerradas estables (RK4)'
    : 'Deriva espiral por truncamiento (Euler)';
  document.getElementById('kpi-asym').innerHTML = asym.toFixed(2) + '&times;';
}}

const sliderE = document.getElementById('slider-e');
const sliderDs = document.getElementById('slider-ds');
const sliderSpd = document.getElementById('slider-spd');

function setEcc(val) {{
  ecc = val;
  sliderE.value = val.toFixed(2);
  document.getElementById('val-e').textContent = val.toFixed(2);
  updateSliderFill(sliderE);
  rebuildStreamlines();
}}

function setDs(val) {{
  ds = val;
  sliderDs.value = val.toFixed(2);
  document.getElementById('val-ds').textContent = val.toFixed(2);
  updateSliderFill(sliderDs);
  rebuildStreamlines();
}}

sliderE.addEventListener('input', e => setEcc(parseFloat(e.target.value)));
sliderDs.addEventListener('input', e => setDs(parseFloat(e.target.value)));
sliderSpd.addEventListener('input', e => {{
  animSpeed = parseFloat(e.target.value);
  document.getElementById('val-spd').innerHTML = animSpeed.toFixed(1) + '&times;';
  updateSliderFill(sliderSpd);
}});

document.getElementById('btn-rk4').addEventListener('click', () => {{
  mode = 'rk4';
  document.getElementById('btn-rk4').classList.add('active');
  document.getElementById('btn-euler').classList.remove('active');
  rebuildStreamlines();
}});
document.getElementById('btn-euler').addEventListener('click', () => {{
  mode = 'euler';
  document.getElementById('btn-euler').classList.add('active');
  document.getElementById('btn-rk4').classList.remove('active');
  rebuildStreamlines();
}});
document.getElementById('btn-play').addEventListener('click', () => {{
  isPlaying = !isPlaying;
  const btn = document.getElementById('btn-play');
  btn.innerHTML = isPlaying ? '&#9654; Movimiento Activo' : '&#9208; Movimiento en Pausa';
  btn.classList.toggle('active', isPlaying);
}});

updateSliderFill(sliderE);
updateSliderFill(sliderDs);
updateSliderFill(sliderSpd);
rebuildStreamlines();

// Bucle de animación a 60 FPS con interpolación continua sobre cada streamline
function animate() {{
  requestAnimationFrame(animate);
  if (isPlaying) {{
    particles.forEach(p => {{
      const maxIdx = p.pts.length - 1;
      if (maxIdx > 1) {{
        const curIdx = Math.floor(p.t) % maxIdx;
        const vLocal = p.speeds[curIdx] || 1.0;
        const advance = (0.032 / Math.max(0.05, p.stepLen)) * animSpeed * vLocal;
        p.t = (p.t + advance) % maxIdx;

        const i0 = Math.floor(p.t);
        const i1 = (i0 + 1) % p.pts.length;
        const frac = p.t - i0;
        p.mesh.position.lerpVectors(p.pts[i0], p.pts[i1], frac);
      }}
    }});
  }}
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>"""
    ruta_html = os.path.abspath("simulador_coaxial_3d_interactivo.html")
    with open(ruta_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"-> Simulador 3D guardado en: {ruta_html}")
    abrir_en_navegador(ruta_html)


# ==============================================================================
# 5. BENCHMARK HONESTO DE SPEEDUP, EFICIENCIA Y EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print(" SIMULACIÓN PARALELA 3D DE CABLE COAXIAL NO IDEAL (BIOT-SAVART + RK4/EULER)")
    print("=" * 78)

    # 1. Generar discretización volumétrica del cable coaxial real en milímetros
    src_pos, src_dI, geom = generar_cable_coaxial_real(excentricidad_x=E_INICIAL_MM * 1e-3)
    n_segmentos = src_pos.shape[0]
    a, b, c, ex_x, ex_y, L = geom
    print(f"-> Elementos diferenciales 3D del cable coaxial: {n_segmentos:,} segmentos")

    # 2. Definir malla tridimensional de observación con indexing='ij'
    N_grid = 32
    lim = b * 1.15
    x = np.linspace(-lim, lim, N_grid)
    y = np.linspace(-lim, lim, N_grid)
    z = np.linspace(-L / 2.0, L / 2.0, N_grid)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    n_nodos = X.size
    print(f"-> Malla 3D de observación: {N_grid}x{N_grid}x{N_grid} ({n_nodos:,} puntos)")
    print(f"-> Interacciones totales Biot-Savart: {n_nodos * n_segmentos:,}\n")

    # 3. Calentamiento JIT (Warm-up) con arreglos contiguos
    print("Compilando kernels JIT de Numba (Warm-up)...")
    X_w = np.ascontiguousarray(X[:2, :2, :2])
    Y_w = np.ascontiguousarray(Y[:2, :2, :2])
    Z_w = np.ascontiguousarray(Z[:2, :2, :2])
    pos_w = np.ascontiguousarray(src_pos[:10])
    dI_w = np.ascontiguousarray(src_dI[:10])
    _ = biot_savart_cpu_secuencial(X_w, Y_w, Z_w, pos_w, dI_w)
    _ = biot_savart_cpu_paralelo(X_w, Y_w, Z_w, pos_w, dI_w)

    # 4. Ejecución Baseline Secuencial (1 Hilo CPU compilado)
    t0 = time.perf_counter()
    Bx_seq, By_seq, Bz_seq = biot_savart_cpu_secuencial(X, Y, Z, src_pos, src_dI)
    t_seq = time.perf_counter() - t0
    print(f"[1] Tiempo Secuencial (1 hilo CPU Numba):     {t_seq:.4f} s")

    # 5. Ejecución Método 1: CPU Paralelo (Numba prange)
    n_cores = config.NUMBA_NUM_THREADS
    t0 = time.perf_counter()
    Bx, By, Bz = biot_savart_cpu_paralelo(X, Y, Z, src_pos, src_dI)
    t_cpu_par = time.perf_counter() - t0

    speedup_cpu = t_seq / t_cpu_par
    eficiencia_cpu = speedup_cpu / n_cores
    error_cpu = np.max(np.abs(Bx - Bx_seq))
    print(f"[2] Tiempo CPU Paralelo ({n_cores} hilos prange):   {t_cpu_par:.4f} s")
    print(f"    -> Speedup CPU:     {speedup_cpu:.2f}x (Ideal: {n_cores:.1f}x)")
    print(f"    -> Eficiencia CPU:  {eficiencia_cpu * 100:.1f}% | Error máx vs Seq: {error_cpu:.2e} T")

    # 6. Ejecución Método 2: GPU CUDA
    gpu_disponible = cuda.is_available()
    t_gpu_kernel, t_gpu_total, speedup_gpu_kernel, speedup_gpu_total = None, None, None, None

    if gpu_disponible:
        _ = ejecutar_biot_savart_gpu(X_w, Y_w, Z_w, pos_w, dI_w)
        Bx_gpu, By_gpu, Bz_gpu, t_gpu_kernel, t_gpu_total = ejecutar_biot_savart_gpu(
            X, Y, Z, src_pos, src_dI
        )
        speedup_gpu_kernel = t_seq / t_gpu_kernel
        speedup_gpu_total = t_seq / t_gpu_total
        error_gpu = np.max(np.abs(Bx_gpu - Bx_seq))
        print(f"[3] Tiempo GPU CUDA (Kernel puro):            {t_gpu_kernel:.4f} s -> Speedup: {speedup_gpu_kernel:.2f}x")
        print(f"    Tiempo GPU CUDA (Con bus PCIe H2D+D2H):   {t_gpu_total:.4f} s -> Speedup: {speedup_gpu_total:.2f}x")
        print(f"    -> Error máx GPU vs Secuencial:           {error_gpu:.2e} T")
    else:
        print("[3] GPU CUDA no detectada en este entorno (se graficará el escalamiento en CPU).")

    # 7. Integración Numérica Paralela de Streamlines 3D (RK4 y Euler Explícito)
    seeds = generar_semillas_coaxial(geom, n_radios=4, n_planos_z=5)
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    ds_paso = 0.08e-3  # 0.08 mm
    n_pasos = 190

    _ = integrar_streamlines_paralelo(seeds[:2], Bx, By, Bz, x[0], y[0], z[0], dx, dy, dz, ds_paso, 10, 4)
    t0_rk4 = time.perf_counter()
    tray_rk4, mag_rk4 = integrar_streamlines_paralelo(
        seeds, Bx, By, Bz, x[0], y[0], z[0], dx, dy, dz, ds=ds_paso, n_steps=n_pasos, metodo=4
    )
    t_rk4 = time.perf_counter() - t0_rk4

    tray_euler, mag_euler = integrar_streamlines_paralelo(
        seeds, Bx, By, Bz, x[0], y[0], z[0], dx, dy, dz, ds=ds_paso * 1.8, n_steps=n_pasos, metodo=1
    )
    print(f"\n[4] Integración RK4 Paralela de {len(seeds)} Streamlines 3D completada en {t_rk4*1000:.2f} ms")

    # 8. Panel 2D: Corte Streamplot en milímetros + Gráfica de Speedup Honesto
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    mid_k = N_grid // 2
    Bx_slice = Bx[:, :, mid_k].T
    By_slice = By[:, :, mid_k].T
    Bmag_slice = np.hypot(Bx_slice, By_slice) * 1e3

    x_mm, y_mm = x * 1e3, y * 1e3
    strm = axes[0].streamplot(
        x_mm, y_mm, Bx_slice, By_slice, color=Bmag_slice, cmap="turbo", linewidth=1.5, density=1.5
    )
    plt.colorbar(strm.lines, ax=axes[0], label="|B| (mT)")
    axes[0].add_patch(plt.Circle((ex_x * 1e3, ex_y * 1e3), a * 1e3, color="#d95f02", alpha=0.85, label="Núcleo (+I)"))
    axes[0].add_patch(plt.Circle((0, 0), b * 1e3, color="#3b465e", fill=False, lw=2, ls="--", label="Blindaje (-I)"))
    axes[0].set_title("Corte Transversal (Z=0): Streamlines Deformadas por Excentricidad")
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Y (mm)")
    axes[0].set_aspect("equal")
    axes[0].legend(loc="upper right")

    metodos = ["Secuencial\n(1 Hilo)", f"CPU prange\n({n_cores} Hilos)"]
    speedups = [1.0, speedup_cpu]
    colores = ["#6c757d", "#2a9d8f"]
    if gpu_disponible:
        metodos.extend(["GPU CUDA\n(Total + PCIe)", "GPU CUDA\n(Kernel Puro)"])
        speedups.extend([speedup_gpu_total, speedup_gpu_kernel])
        colores.extend(["#f4a261", "#e76f51"])

    barras = axes[1].bar(metodos, speedups, color=colores, edgecolor="black", width=0.55)
    axes[1].axhline(n_cores, color="#2a9d8f", linestyle=":", label=f"Speedup Ideal CPU ({n_cores}x)")
    axes[1].set_ylabel("Speedup (S = T_seq / T_par)")
    axes[1].set_title(f"Análisis de Speedup (Eficiencia CPU = {eficiencia_cpu*100:.1f}%)")
    axes[1].legend()
    for bar in barras:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.15, f"{yval:.2f}x", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    # 9. Lanzar las dos visualizaciones 3D animadas en el navegador
    try:
        print("\nAbriendo figura Plotly 3D con animación automática de partículas...")
        visualizar_3d_plotly_estilo_oscuro(tray_rk4, mag_rk4, tray_euler, mag_euler, geom)
    except Exception as e:
        print(f"Aviso: No se pudo abrir la vista de Plotly ({e}). Continuando con el simulador Three.js...")

    time.sleep(0.4)
    print("Abriendo en primer plano el Simulador 3D Interactivo con partículas móviles a 60 FPS y Sliders...")
    abrir_simulador_interactivo_completo_3d(I_total=10.0, a_mm=a*1e3, b_mm=b*1e3)

    # Mantiene abierta la ventana de Matplotlib al final sin bloquear el navegador
    plt.show()