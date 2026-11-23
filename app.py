import math
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageOps
from ultralytics import YOLO

st.set_page_config(page_title="Super ALice - Diagnóstico Vial", layout="wide")

@st.cache_resource
def cargar_modelo():
    return YOLO("best.pt")

st.title("Super ALice · Diagnóstico Vial")

# --- BARRA LATERAL ---
st.sidebar.header("Parámetros de Captura")
altura_cm = st.sidebar.number_input("Altura de la cámara al suelo (cm)", min_value=20.0, max_value=400.0, value=100.0, step=5.0)
espesor_bache_cm = st.sidebar.number_input("Espesor de bacheo (cm)", min_value=1.0, max_value=30.0, value=5.0, step=0.5)
desperdicio_pct = st.sidebar.slider("Desperdicio en obra (%)", min_value=0, max_value=30, value=10, step=5)
conf_threshold = st.sidebar.slider("Umbral de confianza", min_value=0.05, max_value=1.00, value=0.30, step=0.05)
factor_desp = 1.0 + (desperdicio_pct / 100.0)

try:
    model = cargar_modelo()
except Exception as e:
    st.error(f"No se pudo cargar 'best.pt': {e}")
    st.stop()

def extraer_datos_opticos(image_raw, dist_cm):
    exif = image_raw.getexif()
    marca = ""
    modelo = ""
    f35 = None

    if exif:
        marca = str(exif.get(271, "")).strip()
        modelo = str(exif.get(272, "")).strip()
        f35_val = exif.get(41989)
        if f35_val is None and 0x8769 in exif:
            exif_ifd = exif.get_ifd(0x8769)
            f35_val = exif_ifd.get(41989)
        if f35_val is not None:
            try:
                val = float(f35_val)
                if val > 0:
                    f35 = val
            except (TypeError, ValueError):
                pass

    # Corregir orientación automática de fotografías móviles antes de medir píxeles
    image_orientada = ImageOps.exif_transpose(image_raw).convert("RGB")
    w_px, h_px = image_orientada.size
    lado_mayor_px = max(w_px, h_px)

    tiene_exif = f35 is not None
    f_usada = f35 if tiene_exif else 26.0  # Lente gran angular estándar móvil (26 mm eq.)
    cm_por_px = (dist_cm * 36.0) / (f_usada * lado_mayor_px)

    dispositivo = f"{marca} {modelo}".strip() if (marca or modelo) else "Perfil óptico estándar"
    return image_orientada, cm_por_px, f_usada, tiene_exif, dispositivo, lado_mayor_px

def formatear_presentacion_practica(cantidad, unidad):
    if unidad == "L":
        if cantidad < 19.0:
            capacidad = 1.0
            botellas = max(1, math.ceil(cantidad / capacidad))
            etiqueta = "botella de 1 L" if botellas == 1 else "botellas de 1 L"
            presentacion = f"{botellas} {etiqueta}"
            fraccion = cantidad / capacidad
            consumo_real = f"{fraccion:.3f} de la botella de 1 L" if fraccion < 1.0 else f"{fraccion:.2f} botellas de 1 L"
            formula_envase = rf"\text{{Consumo real}} = \frac{{{cantidad:.3f}\text{{ L}}}}{{1\text{{ L/botella}}}} = \mathbf{{{consumo_real}}} \quad \Rightarrow \quad \mathbf{{{presentacion}}}"
            return presentacion, consumo_real, formula_envase
        else:
            capacidad = 19.0
            cubetas = math.ceil(cantidad / capacidad)
            etiqueta = "cubeta de 19 L" if cubetas == 1 else "cubetas de 19 L"
            presentacion = f"{cubetas} {etiqueta}"
            fraccion = cantidad / capacidad
            consumo_real = f"{fraccion:.2f} cubetas de 19 L"
            formula_envase = rf"\text{{Consumo real}} = \frac{{{cantidad:.3f}\text{{ L}}}}{{19\text{{ L/cubeta}}}} = \mathbf{{{consumo_real}}} \quad \Rightarrow \quad \mathbf{{{presentacion}}}"
            return presentacion, consumo_real, formula_envase

    elif unidad == "kg":
        capacidad = 25.0
        costales = max(1, math.ceil(cantidad / capacidad))
        etiqueta = "costal de 25 kg" if costales == 1 else "costales de 25 kg"
        presentacion = f"{costales} {etiqueta}"
        fraccion = cantidad / capacidad
        consumo_real = f"{fraccion:.2f} del costal de 25 kg" if fraccion < 1.0 else f"{fraccion:.2f} costales de 25 kg"
        formula_envase = rf"\text{{Consumo real}} = \frac{{{cantidad:.3f}\text{{ kg}}}}{{25\text{{ kg/costal}}}} = \mathbf{{{consumo_real}}} \quad \Rightarrow \quad \mathbf{{{presentacion}}}"
        return presentacion, consumo_real, formula_envase

    return f"{cantidad:.3f} {unidad}", f"{cantidad:.3f} {unidad}", ""

def calcular_dosificacion(nombre_clase, area_m2, espesor_cm, f_desp):
    clave = nombre_clase.lower().strip()
    espesor_m = espesor_cm / 100.0

    if "bache" in clave:
        mezcla_kg = round(area_m2 * espesor_m * 2350.0 * f_desp, 3)
        emulsion_l = round(area_m2 * 0.80 * f_desp, 3)
        return {
            "accion": "Cajeo, riego de liga y bacheo asfáltico",
            "insumos": [
                {
                    "nombre": "Mezcla asfáltica",
                    "cantidad": mezcla_kg,
                    "unidad": "kg",
                    "explicacion": f"Se multiplica la superficie del bache por el espesor ({espesor_cm:.1f} cm = {espesor_m:.3f} m), la densidad compactada (2,350 kg/m³) y el desperdicio.",
                    "formula": r"M_{\text{mezcla}} = A_{\text{total}} \times \left(\frac{e_{\text{cm}}}{100}\right) \times 2350 \times F_{\text{desp}}",
                    "resolucion": rf"M_{{\text{{mezcla}}}} = {area_m2:.4f} \times \left(\frac{{{espesor_cm:.1f}}}{{100}}\right) \times 2350 \times {f_desp:.2f} = \mathbf{{{mezcla_kg:.3f}\text{{ kg}}}}"
                },
                {
                    "nombre": "Emulsión de liga",
                    "cantidad": emulsion_l,
                    "unidad": "L",
                    "explicacion": "Se aplica emulsión asfáltica de adherencia a razón de 0.80 L/m² sobre la superficie limpia.",
                    "formula": r"V_{\text{liga}} = A_{\text{total}} \times 0.80 \times F_{\text{desp}}",
                    "resolucion": rf"V_{{\text{{liga}}}} = {area_m2:.4f} \times 0.80 \times {f_desp:.2f} = \mathbf{{{emulsion_l:.3f}\text{{ L}}}}"
                }
            ]
        }

    elif "alligator" in clave or "grieta" in clave:
        sellador_l = round(area_m2 * 1.20 * f_desp, 3)
        slurry_l = round(area_m2 * 1.50 * f_desp, 3)
        return {
            "accion": "Sello de grietas y mortero asfáltico (slurry)",
            "insumos": [
                {
                    "nombre": "Sellador de grietas",
                    "cantidad": sellador_l,
                    "unidad": "L",
                    "explicacion": "Las grietas interconectadas requieren 1.20 L/m² de sellador elastomérico para evitar filtración de agua.",
                    "formula": r"V_{\text{sellador}} = A_{\text{total}} \times 1.20 \times F_{\text{desp}}",
                    "resolucion": rf"V_{{\text{{sellador}}}} = {area_m2:.4f} \times 1.20 \times {f_desp:.2f} = \mathbf{{{sellador_l:.3f}\text{{ L}}}}"
                },
                {
                    "nombre": "Mortero slurry",
                    "cantidad": slurry_l,
                    "unidad": "L",
                    "explicacion": "Se extiende mortero asfáltico tipo slurry a razón de 1.50 L/m² para estabilizar la zona agrietada.",
                    "formula": r"V_{\text{slurry}} = A_{\text{total}} \times 1.50 \times F_{\text{desp}}",
                    "resolucion": rf"V_{{\text{{slurry}}}} = {area_m2:.4f} \times 1.50 \times {f_desp:.2f} = \mathbf{{{slurry_l:.3f}\text{{ L}}}}"
                }
            ]
        }

    elif "oxidacion" in clave or "oxidación" in clave or "uv" in clave:
        fog_l = round(area_m2 * 0.70 * f_desp, 3)
        arena_kg = round(area_m2 * 4.00 * f_desp, 3)
        return {
            "accion": "Riego rejuvenecedor (Fog Seal) y arenado",
            "insumos": [
                {
                    "nombre": "Emulsión Fog Seal",
                    "cantidad": fog_l,
                    "unidad": "L",
                    "explicacion": "Rehidrata el asfalto oxidado por rayos UV a razón de 0.70 L/m².",
                    "formula": r"V_{\text{fog}} = A_{\text{total}} \times 0.70 \times F_{\text{desp}}",
                    "resolucion": rf"V_{{\text{{fog}}}} = {area_m2:.4f} \times 0.70 \times {f_desp:.2f} = \mathbf{{{fog_l:.3f}\text{{ L}}}}"
                },
                {
                    "nombre": "Arena de secado",
                    "cantidad": arena_kg,
                    "unidad": "kg",
                    "explicacion": "Se esparce arena fina a razón de 4.00 kg/m² para secar el exceso de emulsión.",
                    "formula": r"M_{\text{arena}} = A_{\text{total}} \times 4.00 \times F_{\text{desp}}",
                    "resolucion": rf"M_{{\text{{arena}}}} = {area_m2:.4f} \times 4.00 \times {f_desp:.2f} = \mathbf{{{arena_kg:.3f}\text{{ kg}}}}"
                }
            ]
        }

    elif "hidrocarburo" in clave:
        deseng_l = round(area_m2 * 0.50 * f_desp, 3)
        barrera_l = round(area_m2 * 0.90 * f_desp, 3)
        return {
            "accion": "Lavado desengrasante y sello barrera",
            "insumos": [
                {
                    "nombre": "Desengrasante industrial",
                    "cantidad": deseng_l,
                    "unidad": "L",
                    "explicacion": "Neutraliza derrames de aceite o combustible a razón de 0.50 L/m².",
                    "formula": r"V_{\text{deseng}} = A_{\text{total}} \times 0.50 \times F_{\text{desp}}",
                    "resolucion": rf"V_{{\text{{deseng}}}} = {area_m2:.4f} \times 0.50 \times {f_desp:.2f} = \mathbf{{{deseng_l:.3f}\text{{ L}}}}"
                },
                {
                    "nombre": "Sellador barrera",
                    "cantidad": barrera_l,
                    "unidad": "L",
                    "explicacion": "Protege la carpeta contra nuevos solventes a razón de 0.90 L/m².",
                    "formula": r"V_{\text{barrera}} = A_{\text{total}} \times 0.90 \times F_{\text{desp}}",
                    "resolucion": rf"V_{{\text{{barrera}}}} = {area_m2:.4f} \times 0.90 \times {f_desp:.2f} = \mathbf{{{barrera_l:.3f}\text{{ L}}}}"
                }
            ]
        }

    else:
        sellador_gen = round(area_m2 * 1.00 * f_desp, 3)
        return {
            "accion": "Sellado superficial preventivo",
            "insumos": [
                {
                    "nombre": "Sellador asfáltico",
                    "cantidad": sellador_gen,
                    "unidad": "L",
                    "explicacion": "Capa protectora estándar de 1.00 L/m² más desperdicio.",
                    "formula": r"V_{\text{sellador}} = A_{\text{total}} \times 1.00 \times F_{\text{desp}}",
                    "resolucion": rf"V_{{\text{{sellador}}}} = {area_m2:.4f} \times 1.00 \times {f_desp:.2f} = \mathbf{{{sellador_gen:.3f}\text{{ L}}}}"
                }
            ]
        }

# --- CARGA DE IMAGEN ---
uploaded_file = st.file_uploader("Toma o sube una fotografía cenital del pavimento", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_raw = Image.open(uploaded_file)
    image, cm_por_pixel_auto, f_usada, tiene_exif, dispositivo, lado_mayor_px = extraer_datos_opticos(image_raw, altura_cm)

    # Permitir ajuste fino manual en barra lateral si el usuario desea sobrescribir
    st.sidebar.markdown("---")
    if tiene_exif:
        st.sidebar.caption(f"Óptica EXIF: {dispositivo} ({f_usada:.0f} mm eq.)")
    else:
        st.sidebar.caption("Sin metadatos EXIF (usando lente estándar 26 mm eq.)")

    cm_por_pixel = st.sidebar.number_input(
        "Escala calibrada (cm/px)",
        min_value=0.005,
        max_value=5.000,
        value=round(cm_por_pixel_auto, 4),
        step=0.005,
        format="%.4f"
    )

    img_array = np.array(image)
    results = model.predict(source=img_array, conf=conf_threshold, verbose=False)
    result = results[0]

    plotted_bgr = result.plot(conf=False)
    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Imagen Original", use_container_width=True)
    with col2:
        st.image(plotted_rgb, caption="Zonas Detectadas", use_container_width=True)

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        st.info("Sin daños detectados por encima del umbral configurado.")
    else:
        muestras = []
        for idx, box in enumerate(boxes, start=1):
            cls_id = int(box.cls[0].item())
            nombre_clase = model.names[cls_id]
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            ancho_px = x2 - x1
            largo_px = y2 - y1
            ancho_cm = ancho_px * cm_por_pixel
            largo_cm = largo_px * cm_por_pixel
            area_cm2 = ancho_cm * largo_cm
            area_m2 = area_cm2 / 10000.0

            muestras.append({
                "num": idx,
                "muestra": f"Muestra {idx}",
                "clase": nombre_clase,
                "conf": conf,
                "ancho_px": ancho_px,
                "largo_px": largo_px,
                "area_m2": area_m2
            })

        df_muestras = pd.DataFrame(muestras)

        df_consolidado = (
            df_muestras.groupby("clase", as_index=False)
            .agg(
                zonas=("num", "count"),
                area_m2_total=("area_m2", "sum"),
            )
            .sort_values(by="area_m2_total", ascending=False)
        )

        # --- RESULTADOS MINIMALISTAS ---
        st.markdown("---")
        st.subheader("Resultados")

        resumen_danos = []
        items_materiales = []

        for _, row in df_consolidado.iterrows():
            patologia = row["clase"]
            area_tot = round(row["area_m2_total"], 4)
            dosif = calcular_dosificacion(patologia, area_tot, espesor_bache_cm, factor_desp)

            resumen_danos.append({
                "Patología": patologia,
                "Superficie Total": f"{area_tot:.4f} m²",
                "Solución": dosif["accion"]
            })

            for ins in dosif["insumos"]:
                presentacion, consumo_real, formula_envase = formatear_presentacion_practica(ins["cantidad"], ins["unidad"])
                items_materiales.append({
                    "material": ins["nombre"],
                    "cantidad_exacta": f"{ins['cantidad']:.3f} {ins['unidad']}",
                    "presentacion": presentacion,
                    "consumo_real": consumo_real,
                    "formula_envase": formula_envase,
                    "patologia": patologia,
                    "area_tot": area_tot,
                    "explicacion": ins["explicacion"],
                    "formula": ins["formula"],
                    "resolucion": ins["resolucion"]
                })

        st.dataframe(pd.DataFrame(resumen_danos), use_container_width=True, hide_index=True)

        st.caption("Haz clic sobre cualquier resultado para ver la fracción real utilizada y su procedimiento:")

        cols_metricas = st.columns(min(len(items_materiales), 4))
        for i, item in enumerate(items_materiales):
            with cols_metricas[i % len(cols_metricas)]:
                with st.container(border=True):
                    st.markdown(f"**{item['material']}**")
                    contenedor_clic = st.popover(f"**{item['presentacion']}**", use_container_width=True) if hasattr(st, "popover") else st.expander(item["presentacion"], expanded=False)
                    with contenedor_clic:
                        st.markdown(f"**Consumo real:** {item['consumo_real']} (`{item['cantidad_exacta']}`)")
                        st.write(item["explicacion"])

                        st.markdown(f"**1. Calibración óptica ({dispositivo})**")
                        st.latex(
                            rf"E_{{\text{{cenital}}}} = \frac{{D_{{\text{{cm}}}} \times 36\text{{ mm}}}}{{f_{{35}} \times L_{{\text{{px}}}}}} = \frac{{{altura_cm:.0f} \times 36}}{{{f_usada:.1f} \times {lado_mayor_px}}} = \mathbf{{{cm_por_pixel:.4f}\text{{ cm/px}}}}"
                        )

                        st.markdown(f"**2. Superficie de `{item['patologia']}`**")
                        sub = [m for m in muestras if m["clase"] == item["patologia"]]
                        for m in sub:
                            st.latex(
                                rf"\text{{{m['muestra']}: }} A_{{{m['num']}}} = \frac{{({m['ancho_px']:.1f} \times {cm_por_pixel:.4f}) \times ({m['largo_px']:.1f} \times {cm_por_pixel:.4f})}}{{10,000}} = {m['area_m2']:.4f}\text{{ m}}^2"
                            )
                        if len(sub) > 1:
                            suma_str = " + ".join([f"{m['area_m2']:.4f}" for m in sub])
                            st.latex(rf"A_{{\text{{total}}}} = {suma_str} = \mathbf{{{item['area_tot']:.4f}\text{{ m}}^2}}")

                        st.markdown(f"**3. Cantidad exacta (+{desperdicio_pct}% desperdicio)**")
                        st.latex(item["formula"])
                        st.latex(item["resolucion"])

                        if item["formula_envase"]:
                            st.markdown("**4. Fracción real del envase**")
                            st.latex(item["formula_envase"])

        # Descarga CSV
        df_export = pd.DataFrame([
            {
                "Material": x["material"],
                "Envases a Llevar": x["presentacion"],
                "Consumo Real del Envase": x["consumo_real"],
                "Cantidad Neta": x["cantidad_exacta"],
                "Patología": x["patologia"]
            }
            for x in items_materiales
        ])
        st.download_button(
            label="Descargar resultados (CSV)",
            data=df_export.to_csv(index=False).encode("utf-8"),
            file_name="resultados_super_alice.csv",
            mime="text/csv"
        )

        # --- PESTAÑA CONTRAÍBLE: FIABILIDAD POR MUESTRA ---
        with st.expander("Fiabilidad de detección por muestra", expanded=False):
            df_fiabilidad = pd.DataFrame({
                "Muestra": df_muestras["muestra"],
                "Patología": df_muestras["clase"],
                "Medidas Crudas (px)": [f"{m['ancho_px']:.0f} × {m['largo_px']:.0f}" for m in muestras],
                "Fiabilidad": [f"{m['conf'] * 100:.1f}%" for m in muestras]
            })
            st.dataframe(df_fiabilidad, use_container_width=True, hide_index=True)
