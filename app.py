import io
import math
import uuid
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageOps
from fpdf import FPDF
from ultralytics import YOLO

st.set_page_config(page_title="Super ALice - Diagnóstico Vial", layout="wide")

@st.cache_resource
def cargar_modelo():
    return YOLO("best.pt")

st.title("Super ALice · Diagnóstico Vial")

# --- BARRA LATERAL: PARÁMETROS DE CAPTURA ---
st.sidebar.header("Parámetros de Captura")
altura_cm = st.sidebar.number_input("Altura de la cámara al suelo (cm)", min_value=20.0, max_value=400.0, value=100.0, step=5.0)
espesor_bache_cm = st.sidebar.number_input("Espesor de bacheo (cm)", min_value=1.0, max_value=30.0, value=5.0, step=0.5)
desperdicio_pct = st.sidebar.slider("Desperdicio en obra (%)", min_value=0, max_value=30, value=10, step=5)
conf_threshold = st.sidebar.slider("Umbral de confianza", min_value=0.05, max_value=1.00, value=0.30, step=0.05)
factor_desp = 1.0 + (desperdicio_pct / 100.0)

st.sidebar.markdown("---")
usar_escala_manual = st.sidebar.checkbox("Forzar escala manual (cm/px)", value=False)
escala_manual_val = 0.15
if usar_escala_manual:
    escala_manual_val = st.sidebar.number_input(
        "Escala manual (cm/px)", min_value=0.005, max_value=5.000, value=0.1500, step=0.005, format="%.4f"
    )

# --- BARRA LATERAL: COSTOS UNITARIOS CONTRAÍBLES ---
st.sidebar.markdown("---")
with st.sidebar.expander("Costos Unitarios (MXN)", expanded=False):
    st.caption("Precios por presentación comercial:")
    costos_unitarios = {
        ("Mezcla asfáltica", "costal"): st.number_input("Mezcla asfáltica · Costal 25 kg ($)", min_value=0.0, value=220.0, step=10.0),
        ("Emulsión de liga", "botella"): st.number_input("Emulsión liga · Botella 1 L ($)", min_value=0.0, value=45.0, step=5.0),
        ("Emulsión de liga", "cubeta"): st.number_input("Emulsión liga · Cubeta 19 L ($)", min_value=0.0, value=680.0, step=20.0),
        ("Sellador de grietas", "botella"): st.number_input("Sellador grietas · Botella 1 L ($)", min_value=0.0, value=140.0, step=10.0),
        ("Sellador de grietas", "cubeta"): st.number_input("Sellador grietas · Cubeta 19 L ($)", min_value=0.0, value=2200.0, step=50.0),
        ("Mortero slurry", "botella"): st.number_input("Mortero slurry · Botella 1 L ($)", min_value=0.0, value=65.0, step=5.0),
        ("Mortero slurry", "cubeta"): st.number_input("Mortero slurry · Cubeta 19 L ($)", min_value=0.0, value=950.0, step=25.0),
        ("Emulsión Fog Seal", "botella"): st.number_input("Emulsión Fog Seal · Botella 1 L ($)", min_value=0.0, value=55.0, step=5.0),
        ("Emulsión Fog Seal", "cubeta"): st.number_input("Emulsión Fog Seal · Cubeta 19 L ($)", min_value=0.0, value=820.0, step=25.0),
        ("Arena de secado", "costal"): st.number_input("Arena secado · Costal 25 kg ($)", min_value=0.0, value=60.0, step=5.0),
        ("Desengrasante industrial", "botella"): st.number_input("Desengrasante · Botella 1 L ($)", min_value=0.0, value=85.0, step=5.0),
        ("Desengrasante industrial", "cubeta"): st.number_input("Desengrasante · Cubeta 19 L ($)", min_value=0.0, value=1300.0, step=50.0),
        ("Sellador barrera", "botella"): st.number_input("Sellador barrera · Botella 1 L ($)", min_value=0.0, value=160.0, step=10.0),
        ("Sellador barrera", "cubeta"): st.number_input("Sellador barrera · Cubeta 19 L ($)", min_value=0.0, value=2500.0, step=50.0),
        ("Sellador asfáltico", "botella"): st.number_input("Sellador general · Botella 1 L ($)", min_value=0.0, value=90.0, step=5.0),
        ("Sellador asfáltico", "cubeta"): st.number_input("Sellador general · Cubeta 19 L ($)", min_value=0.0, value=1400.0, step=50.0),
    }

try:
    model = cargar_modelo()
except Exception as e:
    st.error(f"No se pudo cargar 'best.pt': {e}")
    st.stop()

# --- UTILIDADES EXIF (ÓPTICA + GPS) ---
def _dms_a_decimal(dms, ref):
    try:
        grados = float(dms[0])
        minutos = float(dms[1])
        segundos = float(dms[2])
        decimal = grados + (minutos / 60.0) + (segundos / 3600.0)
        if str(ref).strip().upper() in ("S", "W"):
            decimal = -decimal
        return round(decimal, 6)
    except Exception:
        return None

def extraer_metadatos_imagen(image_raw, dist_cm):
    exif = image_raw.getexif()
    marca = ""
    modelo = ""
    f35 = None
    lat = None
    lon = None

    if exif:
        marca = str(exif.get(271, "")).strip()
        modelo = str(exif.get(272, "")).strip()
        f35_val = exif.get(41989)
        if f35_val is None and 0x8769 in exif:
            try:
                exif_ifd = exif.get_ifd(0x8769)
                f35_val = exif_ifd.get(41989)
            except Exception:
                pass

        if f35_val is not None:
            try:
                val = float(f35_val)
                if val > 0:
                    f35 = val
            except (TypeError, ValueError):
                pass

        # Extraer GPSInfo (Tag 0x8825 / 34853)
        if 0x8825 in exif:
            try:
                gps_ifd = exif.get_ifd(0x8825)
                if 1 in gps_ifd and 2 in gps_ifd and 3 in gps_ifd and 4 in gps_ifd:
                    lat = _dms_a_decimal(gps_ifd[2], gps_ifd[1])
                    lon = _dms_a_decimal(gps_ifd[4], gps_ifd[3])
            except Exception:
                pass

    image_orientada = ImageOps.exif_transpose(image_raw).convert("RGB")
    w_px, h_px = image_orientada.size
    lado_mayor_px = max(w_px, h_px)

    tiene_exif = (f35 is not None) or bool(marca or modelo)
    f_usada = f35 if f35 is not None else 26.0
    cm_por_px_auto = (dist_cm * 36.0) / (f_usada * lado_mayor_px)
    dispositivo = f"{marca} {modelo}".strip() if (marca or modelo) else "Perfil estándar (sin EXIF)"

    return {
        "imagen": image_orientada,
        "cm_por_px_auto": cm_por_px_auto,
        "f_usada": f_usada,
        "tiene_exif": tiene_exif,
        "dispositivo": dispositivo,
        "lado_mayor_px": lado_mayor_px,
        "lat": lat,
        "lon": lon,
    }

# --- PRESENTACIONES COMERCIALES Y COSTOS ---
def formatear_presentacion_y_costo(nombre_material, cantidad, unidad, precios_dict):
    if unidad == "L":
        if cantidad < 19.0:
            capacidad = 1.0
            tipo_envase = "botella"
            num_envases = max(1, math.ceil(cantidad / capacidad))
            etiqueta = "botella de 1 L" if num_envases == 1 else "botellas de 1 L"
            presentacion = f"{num_envases} {etiqueta}"
            fraccion = cantidad / capacidad
            consumo_real = f"{fraccion:.3f} de la botella de 1 L" if fraccion < 1.0 else f"{fraccion:.2f} botellas de 1 L"
            formula_envase = rf"\text{{Consumo real}} = \frac{{{cantidad:.3f}\text{{ L}}}}{{1\text{{ L/botella}}}} = \mathbf{{{consumo_real}}} \quad \Rightarrow \quad \mathbf{{{presentacion}}}"
        else:
            capacidad = 19.0
            tipo_envase = "cubeta"
            num_envases = math.ceil(cantidad / capacidad)
            etiqueta = "cubeta de 19 L" if num_envases == 1 else "cubetas de 19 L"
            presentacion = f"{num_envases} {etiqueta}"
            fraccion = cantidad / capacidad
            consumo_real = f"{fraccion:.2f} cubetas de 19 L"
            formula_envase = rf"\text{{Consumo real}} = \frac{{{cantidad:.3f}\text{{ L}}}}{{19\text{{ L/cubeta}}}} = \mathbf{{{consumo_real}}} \quad \Rightarrow \quad \mathbf{{{presentacion}}}"
    elif unidad == "kg":
        capacidad = 25.0
        tipo_envase = "costal"
        num_envases = max(1, math.ceil(cantidad / capacidad))
        etiqueta = "costal de 25 kg" if num_envases == 1 else "costales de 25 kg"
        presentacion = f"{num_envases} {etiqueta}"
        fraccion = cantidad / capacidad
        consumo_real = f"{fraccion:.2f} del costal de 25 kg" if fraccion < 1.0 else f"{fraccion:.2f} costales de 25 kg"
        formula_envase = rf"\text{{Consumo real}} = \frac{{{cantidad:.3f}\text{{ kg}}}}{{25\text{{ kg/costal}}}} = \mathbf{{{consumo_real}}} \quad \Rightarrow \quad \mathbf{{{presentacion}}}"
    else:
        return {
            "presentacion": f"{cantidad:.3f} {unidad}",
            "consumo_real": f"{cantidad:.3f} {unidad}",
            "formula_envase": "",
            "precio_unitario": 0.0,
            "costo_compra": 0.0,
            "costo_neto": 0.0,
        }

    precio_u = precios_dict.get((nombre_material, tipo_envase), 0.0)
    costo_compra = round(num_envases * precio_u, 2)
    costo_neto = round(fraccion * precio_u, 2)

    return {
        "presentacion": presentacion,
        "consumo_real": consumo_real,
        "formula_envase": formula_envase,
        "precio_unitario": precio_u,
        "costo_compra": costo_compra,
        "costo_neto": costo_neto,
    }

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
                    "explicacion": f"Se multiplica la superficie total por el espesor ({espesor_cm:.1f} cm = {espesor_m:.3f} m), la densidad compactada (2,350 kg/m³) y el desperdicio.",
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

# --- GENERACIÓN DE DICTAMEN TÉCNICO EN PDF ---
def _txt_pdf(texto):
    reemplazos = {
        "²": "2",
        "³": "3",
        "·": "-",
        "—": "-",
        "–": "-",
        "→": "->",
        "×": "x",
        "“": '"',
        "”": '"',
    }
    s = str(texto)
    for k, v in reemplazos.items():
        s = s.replace(k, v)
    return s.encode("latin-1", errors="replace").decode("latin-1")

def _pil_a_bytes_jpeg(pil_img, max_ancho=800):
    img_copy = pil_img.copy()
    if img_copy.width > max_ancho:
        ratio = max_ancho / float(img_copy.width)
        nuevo_alto = int(img_copy.height * ratio)
        img_copy = img_copy.resize((max_ancho, nuevo_alto), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img_copy.save(buf, format="JPEG", quality=82)
    buf.seek(0)
    return buf

def generar_dictamen_pdf(folio, fecha_str, num_fotos, gps_lista, resumen_danos, items_materiales, presupuesto_total, imagenes_lote):
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Encabezado institucional
    pdf.set_fill_color(24, 34, 48)
    pdf.rect(0, 0, 210, 26, style="F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_xy(12, 6)
    pdf.cell(0, 7, _txt_pdf("SUPER ALice - Dictamen Técnico de Inspección Vial"), ln=True)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_x(12)
    pdf.cell(0, 6, _txt_pdf(f"Folio: {folio}   |   Fecha: {fecha_str}   |   Fotografías analizadas: {num_fotos}"), ln=True)

    pdf.set_text_color(30, 30, 30)
    pdf.set_y(31)

    # Parámetros y GPS
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.cell(0, 6, _txt_pdf("1. Datos Generales y Geolocalización"), ln=True)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.cell(
        0, 5.5,
        _txt_pdf(f"Altura cámara: {altura_cm:.0f} cm  |  Espesor bacheo: {espesor_bache_cm:.1f} cm  |  Desperdicio obra: {desperdicio_pct}%"),
        ln=True
    )

    if gps_lista:
        for g in gps_lista:
            url_maps = f"https://www.google.com/maps?q={g['lat']},{g['lon']}"
            pdf.set_text_color(30, 30, 30)
            pdf.cell(65, 5.5, _txt_pdf(f"- {g['etiqueta']}: {g['lat']}, {g['lon']}"), ln=False)
            pdf.set_text_color(15, 90, 180)
            pdf.cell(0, 5.5, _txt_pdf("Abrir en Google Maps"), ln=True, link=url_maps)
        pdf.set_text_color(30, 30, 30)
    else:
        pdf.cell(0, 5.5, _txt_pdf("Ubicación GPS: No disponible en los metadatos de las imágenes."), ln=True)

    pdf.ln(3)

    # Tabla 1: Cuantificación de Patologías
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.cell(0, 6, _txt_pdf("2. Cuantificación Consolidada de Daños"), ln=True)

    pdf.set_fill_color(235, 238, 242)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(45, 7, _txt_pdf("Patología"), border=1, fill=True)
    pdf.cell(35, 7, _txt_pdf("Superficie Total"), border=1, fill=True)
    pdf.cell(106, 7, _txt_pdf("Intervención Técnica Recomendada"), border=1, fill=True, ln=True)

    pdf.set_font("Helvetica", "", 9)
    for row in resumen_danos:
        pdf.cell(45, 6.5, _txt_pdf(row["Patología"]), border=1)
        pdf.cell(35, 6.5, _txt_pdf(row["Superficie Total"]), border=1)
        pdf.cell(106, 6.5, _txt_pdf(row["Solución"]), border=1, ln=True)

    pdf.ln(4)

    # Tabla 2: Requisición y Presupuesto
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.cell(0, 6, _txt_pdf("3. Requisición Comercial y Presupuesto Estimado"), ln=True)

    pdf.set_fill_color(235, 238, 242)
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.cell(44, 7, _txt_pdf("Material"), border=1, fill=True)
    pdf.cell(38, 7, _txt_pdf("Envases a Adquirir"), border=1, fill=True)
    pdf.cell(44, 7, _txt_pdf("Consumo Neto"), border=1, fill=True)
    pdf.cell(30, 7, _txt_pdf("P. Unitario"), border=1, fill=True, align="R")
    pdf.cell(30, 7, _txt_pdf("Subtotal"), border=1, fill=True, align="R", ln=True)

    pdf.set_font("Helvetica", "", 8.5)
    for item in items_materiales:
        pdf.cell(44, 6.5, _txt_pdf(item["material"]), border=1)
        pdf.cell(38, 6.5, _txt_pdf(item["presentacion"]), border=1)
        pdf.cell(44, 6.5, _txt_pdf(item["consumo_real"]), border=1)
        pdf.cell(30, 6.5, _txt_pdf(f"${item['precio_unitario']:,.2f}"), border=1, align="R")
        pdf.cell(30, 6.5, _txt_pdf(f"${item['costo_compra']:,.2f}"), border=1, align="R", ln=True)

    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_fill_color(245, 247, 250)
    pdf.cell(156, 7.5, _txt_pdf("PRESUPUESTO DIRECTO ESTIMADO (MXN)"), border=1, fill=True, align="R")
    pdf.cell(30, 7.5, _txt_pdf(f"${presupuesto_total:,.2f}"), border=1, fill=True, align="R", ln=True)

    pdf.ln(5)

    # Evidencia Visual Comparativa
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.cell(0, 6, _txt_pdf("4. Evidencia Fotográfica Comparativa"), ln=True)
    pdf.ln(1)

    for idx_img, img_info in enumerate(imagenes_lote[:6], start=1):
        if pdf.get_y() > 215:
            pdf.add_page()

        pdf.set_font("Helvetica", "B", 9)
        titulo_foto = f"Fotografía {idx_img}: {img_info['nombre']}  (Escala: {img_info['escala']:.4f} cm/px)"
        pdf.cell(0, 5.5, _txt_pdf(titulo_foto), ln=True)

        y_actual = pdf.get_y()
        buf_orig = _pil_a_bytes_jpeg(img_info["orig"])
        buf_det = _pil_a_bytes_jpeg(img_info["det"])

        ancho_caja = 88
        alto_max = 58
        ratio_asp = img_info["orig"].height / float(img_info["orig"].width)
        alto_img = min(alto_max, ancho_caja * ratio_asp)

        pdf.image(buf_orig, x=12, y=y_actual, w=ancho_caja, h=alto_img)
        pdf.image(buf_det, x=106, y=y_actual, w=ancho_caja, h=alto_img)
        pdf.set_y(y_actual + alto_img + 4)

    # Firma
    if pdf.get_y() > 255:
        pdf.add_page()
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(90, 5, "________________________________________", align="C")
    pdf.cell(10, 5, "")
    pdf.cell(90, 5, "________________________________________", align="C", ln=True)
    pdf.cell(90, 5, _txt_pdf("Supervisión Técnica / Inspección en Campo"), align="C")
    pdf.cell(10, 5, "")
    pdf.cell(90, 5, _txt_pdf("Vo. Bo. Residencia de Obra"), align="C", ln=True)

    salida = pdf.output()
    return bytes(salida) if isinstance(salida, (bytes, bytearray)) else salida.encode("latin-1")

# --- CARGA POR LOTE DE IMÁGENES ---
uploaded_files = st.file_uploader(
    "Toma o sube una o varias fotografías cenitales del pavimento",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    muestras_globales = []
    imagenes_procesadas = []
    fotos_sin_exif = []
    gps_encontrados = []

    for idx_foto, archivo in enumerate(uploaded_files, start=1):
        image_raw = Image.open(archivo)
        meta = extraer_metadatos_imagen(image_raw, altura_cm)
        img_rgb = meta["imagen"]

        escala_activa = escala_manual_val if usar_escala_manual else meta["cm_por_px_auto"]

        if not meta["tiene_exif"] and not usar_escala_manual:
            fotos_sin_exif.append(archivo.name)

        if meta["lat"] is not None and meta["lon"] is not None:
            gps_encontrados.append({
                "etiqueta": f"Foto {idx_foto} ({archivo.name})",
                "lat": meta["lat"],
                "lon": meta["lon"],
            })

        img_array = np.array(img_rgb)
        results = model.predict(source=img_array, conf=conf_threshold, verbose=False)
        result = results[0]

        plotted_bgr = result.plot(conf=False)
        plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
        img_det_pil = Image.fromarray(plotted_rgb)

        imagenes_procesadas.append({
            "num_foto": idx_foto,
            "nombre": archivo.name,
            "orig": img_rgb,
            "det": img_det_pil,
            "escala": escala_activa,
            "f_usada": meta["f_usada"],
            "dispositivo": meta["dispositivo"],
            "lado_mayor_px": meta["lado_mayor_px"],
        })

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for idx_box, box in enumerate(boxes, start=1):
                cls_id = int(box.cls[0].item())
                nombre_clase = model.names[cls_id]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                ancho_px = x2 - x1
                largo_px = y2 - y1
                ancho_cm = ancho_px * escala_activa
                largo_cm = largo_px * escala_activa
                area_m2 = (ancho_cm * largo_cm) / 10000.0

                muestras_globales.append({
                    "num_global": len(muestras_globales) + 1,
                    "foto_label": f"Foto {idx_foto}" if len(uploaded_files) > 1 else "Foto 1",
                    "muestra": f"Muestra {len(muestras_globales) + 1}",
                    "clase": nombre_clase,
                    "conf": conf,
                    "ancho_px": ancho_px,
                    "largo_px": largo_px,
                    "escala": escala_activa,
                    "f_usada": meta["f_usada"],
                    "dispositivo": meta["dispositivo"],
                    "lado_mayor_px": meta["lado_mayor_px"],
                    "area_m2": area_m2,
                })

    # --- ALERTA DE PÉRDIDA DE METADATOS EXIF ---
    if fotos_sin_exif:
        nombres_str = ", ".join(fotos_sin_exif[:3]) + ("..." if len(fotos_sin_exif) > 3 else "")
        st.warning(
            f"**Sin metadatos originales de cámara** en: `{nombres_str}` (posible captura de pantalla o imagen reenviada por WhatsApp). "
            "Se aplicó un perfil óptico estándar (26 mm eq.); para máxima precisión sube la foto original desde la galería o activa *Forzar escala manual*."
        )

    # --- VISOR DE EVIDENCIA FOTOGRÁFICA ---
    if len(imagenes_procesadas) == 1:
        col1, col2 = st.columns(2)
        with col1:
            st.image(imagenes_procesadas[0]["orig"], caption="Imagen Original", use_container_width=True)
        with col2:
            st.image(imagenes_procesadas[0]["det"], caption="Zonas Detectadas", use_container_width=True)
    else:
        st.caption(f"Lote cargado: **{len(imagenes_procesadas)} fotografías** analizadas en conjunto.")
        with st.expander("Ver comparativa fotográfica del lote", expanded=True):
            cols_lote = st.columns(min(len(imagenes_procesadas), 3))
            for i, info_img in enumerate(imagenes_procesadas):
                with cols_lote[i % len(cols_lote)]:
                    st.image(
                        info_img["det"],
                        caption=f"Foto {info_img['num_foto']} · {info_img['nombre']} ({info_img['escala']:.4f} cm/px)",
                        use_container_width=True
                    )

    # --- MAPA GPS SI EXISTEN COORDENADAS ---
    if gps_encontrados:
        with st.expander("Ubicación GPS detectada", expanded=True):
            df_gps = pd.DataFrame(gps_encontrados)
            st.map(df_gps[["lat", "lon"]], zoom=16)
            links_md = "  |  ".join(
                [f"[{g['etiqueta']}](https://www.google.com/maps?q={g['lat']},{g['lon']})" for g in gps_encontrados]
            )
            st.markdown(f"**Google Maps:** {links_md}")

    # --- PROCESAMIENTO DE RESULTADOS CONSOLIDADOS ---
    if not muestras_globales:
        st.info("Sin daños detectados por encima del umbral configurado.")
    else:
        df_muestras = pd.DataFrame(muestras_globales)
        df_consolidado = (
            df_muestras.groupby("clase", as_index=False)
            .agg(
                zonas=("num_global", "count"),
                area_m2_total=("area_m2", "sum"),
            )
            .sort_values(by="area_m2_total", ascending=False)
        )

        st.markdown("---")
        st.subheader("Resultados Consolidados")

        resumen_danos = []
        items_materiales = []
        presupuesto_total = 0.0

        for _, row in df_consolidado.iterrows():
            patologia = row["clase"]
            area_tot = round(row["area_m2_total"], 4)
            dosif = calcular_dosificacion(patologia, area_tot, espesor_bache_cm, factor_desp)

            resumen_danos.append({
                "Patología": patologia,
                "Regiones": int(row["zonas"]),
                "Superficie Total": f"{area_tot:.4f} m²",
                "Solución": dosif["accion"],
            })

            for ins in dosif["insumos"]:
                env = formatear_presentacion_y_costo(ins["nombre"], ins["cantidad"], ins["unidad"], costos_unitarios)
                presupuesto_total += env["costo_compra"]

                items_materiales.append({
                    "material": ins["nombre"],
                    "cantidad_exacta": f"{ins['cantidad']:.3f} {ins['unidad']}",
                    "presentacion": env["presentacion"],
                    "consumo_real": env["consumo_real"],
                    "formula_envase": env["formula_envase"],
                    "precio_unitario": env["precio_unitario"],
                    "costo_compra": env["costo_compra"],
                    "costo_neto": env["costo_neto"],
                    "patologia": patologia,
                    "area_tot": area_tot,
                    "explicacion": ins["explicacion"],
                    "formula": ins["formula"],
                    "resolucion": ins["resolucion"],
                })

        st.dataframe(pd.DataFrame(resumen_danos), use_container_width=True, hide_index=True)

        st.caption("Haz clic sobre cualquier resultado para ver la fracción real utilizada, su costo y el procedimiento:")

        cols_metricas = st.columns(min(len(items_materiales), 4))
        for i, item in enumerate(items_materiales):
            with cols_metricas[i % len(cols_metricas)]:
                with st.container(border=True):
                    st.markdown(f"**{item['material']}**")
                    contenedor_clic = (
                        st.popover(f"**{item['presentacion']}**", use_container_width=True)
                        if hasattr(st, "popover")
                        else st.expander(item["presentacion"], expanded=False)
                    )
                    with contenedor_clic:
                        st.markdown(f"**Consumo real:** {item['consumo_real']} (`{item['cantidad_exacta']}`)")
                        st.markdown(
                            f"**Costo envases:** `${item['costo_compra']:,.2f} MXN` "
                            f"(costo neto proporcional: `${item['costo_neto']:,.2f} MXN`)"
                        )
                        st.write(item["explicacion"])

                        st.markdown(f"**1. Superficie de `{item['patologia']}`**")
                        sub = [m for m in muestras_globales if m["clase"] == item["patologia"]]
                        for m in sub:
                            etiqueta_m = f"{m['foto_label']} · {m['muestra']}" if len(uploaded_files) > 1 else m["muestra"]
                            st.latex(
                                rf"\text{{{etiqueta_m}: }} A_{{{m['num_global']}}} = \frac{{({m['ancho_px']:.1f} \times {m['escala']:.4f}) \times ({m['largo_px']:.1f} \times {m['escala']:.4f})}}{{10,000}} = {m['area_m2']:.4f}\text{{ m}}^2"
                            )
                        if len(sub) > 1:
                            suma_str = " + ".join([f"{m['area_m2']:.4f}" for m in sub])
                            st.latex(rf"A_{{\text{{total}}}} = {suma_str} = \mathbf{{{item['area_tot']:.4f}\text{{ m}}^2}}")

                        st.markdown(f"**2. Cantidad exacta (+{desperdicio_pct}% desperdicio)**")
                        st.latex(item["formula"])
                        st.latex(item["resolucion"])

                        if item["formula_envase"]:
                            st.markdown("**3. Fracción real del envase**")
                            st.latex(item["formula_envase"])

        # --- PRESUPUESTO ESTIMADO TOTAL ---
        st.markdown("---")
        col_pres, col_desc1, col_desc2 = st.columns([2, 1.3, 1.2])
        with col_pres:
            st.metric(
                label="Presupuesto Directo Estimado (Envases Comerciales)",
                value=f"${presupuesto_total:,.2f} MXN",
                delta="Ajusta precios unitarios en la barra lateral",
                delta_color="off"
            )

        folio_actual = f"ALICE-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
        fecha_actual = datetime.now().strftime("%d/%m/%Y %H:%M")

        pdf_bytes = generar_dictamen_pdf(
            folio=folio_actual,
            fecha_str=fecha_actual,
            num_fotos=len(imagenes_procesadas),
            gps_lista=gps_encontrados,
            resumen_danos=resumen_danos,
            items_materiales=items_materiales,
            presupuesto_total=presupuesto_total,
            imagenes_lote=imagenes_procesadas,
        )

        with col_desc1:
            st.write("")
            st.download_button(
                label="Descargar Dictamen Técnico (PDF)",
                data=pdf_bytes,
                file_name=f"Dictamen_{folio_actual}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        df_export = pd.DataFrame([
            {
                "Folio": folio_actual,
                "Material": x["material"],
                "Envases a Llevar": x["presentacion"],
                "Consumo Real del Envase": x["consumo_real"],
                "Cantidad Neta": x["cantidad_exacta"],
                "Precio Unitario (MXN)": x["precio_unitario"],
                "Subtotal (MXN)": x["costo_compra"],
                "Patología": x["patologia"],
                "GPS": f"{gps_encontrados[0]['lat']}, {gps_encontrados[0]['lon']}" if gps_encontrados else "N/D",
            }
            for x in items_materiales
        ])

        with col_desc2:
            st.write("")
            st.download_button(
                label="Descargar Datos (CSV)",
                data=df_export.to_csv(index=False).encode("utf-8"),
                file_name=f"Datos_{folio_actual}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # --- PESTAÑA CONTRAÍBLE: FIABILIDAD POR MUESTRA ---
        with st.expander("Fiabilidad de detección por muestra", expanded=False):
            df_fiabilidad = pd.DataFrame({
                "Origen": df_muestras["foto_label"],
                "Muestra": df_muestras["muestra"],
                "Patología": df_muestras["clase"],
                "Medidas Crudas (px)": [f"{m['ancho_px']:.0f} × {m['largo_px']:.0f}" for m in muestras_globales],
                "Fiabilidad": [f"{m['conf'] * 100:.1f}%" for m in muestras_globales],
            })
            st.dataframe(df_fiabilidad, use_container_width=True, hide_index=True)
