# -*- coding: utf-8 -*-
# =======================================================================================
# PARTE 1/13: Imports, Logging, Timezone Setup (vD+DX v4 - CORREGIDO 25-Abr)
# =======================================================================================

# --- Imports Estándar ---
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import logging
import sys
import warnings
import math
import decimal  # Usado en calculate_grid_levels
import time     # Para medir tiempos
# import copy     # No usado activamente ahora (sensibilidad comentada)
import json     # Para formatear diccionarios en logs/txt
import traceback # Para loguear errores detallados
import os       # Para os.getpid() y gestión de archivos
import sqlite3  # Para guardar trades externamente
from io import StringIO # Para capturar salida de df.info() en logs

# --- Imports para TensorFlow (GPU Check) ---
_tf_available = False # Asumir no disponible por defecto
try:
    # Silenciar logs de TF antes de importar (Nivel 2: solo errores)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    # Establecer nivel de logger de TF a WARNING después de importar
    tf.get_logger().setLevel('WARNING')
    _tf_available = True
except ImportError:
    # Advertir si TF no está instalado
    print("ADVERTENCIA: TensorFlow no está instalado. Funciones de predicción TF no disponibles.", file=sys.stderr)
except Exception as e_tf_import:
    # Capturar otros posibles errores durante la importación de TF
    print(f"ADVERTENCIA: Error importando TensorFlow: {e_tf_import}", file=sys.stderr)

# --- Imports para Optimización (Optuna) ---
try:
    import optuna
    # Opcional: Silenciar logs INFO de Optuna (útil si genera mucha salida)
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    # Error crítico si Optuna no está instalado
    print("ERROR CRITICO: Optuna no está instalado. Ejecuta: pip install optuna", file=sys.stderr)
    # Relanzar para detener la ejecución si Optuna es esencial
    raise ImportError("Dependencia 'optuna' no encontrada.")

# --- Imports para Zonas Horarias ---
_timezone_info_available = False
lima_tz = None
try:
    # Preferir zoneinfo (Python >= 3.9)
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        lima_tz = ZoneInfo("America/Lima") # Reemplaza con tu zona horaria si es diferente
        _timezone_info_available = True
    except ZoneInfoNotFoundError:
        print("Advertencia: Timezone 'America/Lima' no encontrada con zoneinfo.", file=sys.stderr)
except ImportError:
    # Fallback a pytz si zoneinfo no está disponible
    try:
        import pytz
        try:
            lima_tz = pytz.timezone("America/Lima") # Reemplaza con tu zona horaria si es diferente
            _timezone_info_available = True
        except pytz.UnknownTimeZoneError:
            print("Advertencia: Timezone 'America/Lima' no encontrada con pytz.", file=sys.stderr)
    except ImportError:
        # Si ninguno está disponible, se usará la hora local del sistema (naive)
        pass

# --- Configuración del Logging ---
log_level = logging.INFO # Nivel de detalle para logs (DEBUG es el más detallado)
# Nombre del archivo de log (puedes cambiarlo si quieres una nueva versión)
log_filename = "run_debug_vD_SQLITE_DX_v5_runonce_false.log" # NUEVO NOMBRE SUGERIDO

# Configurar logging básico
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(process)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'), # Sobrescribir
        logging.StreamHandler(sys.stdout) # Mostrar en consola
    ],
    force=True # Forzar reconfiguración
)

# --- Logs Iniciales de Información del Entorno ---
logging.info(f"--- Script Iniciado (PID: {os.getpid()}) ---")
logging.info(f"Python Version: {sys.version.split()[0]}")
logging.info(f"Backtrader Version: {bt.__version__}")
logging.info(f"Pandas Version: {pd.__version__}")
logging.info(f"Numpy Version: {np.__version__}")
logging.info(f"Optuna versión {optuna.__version__} disponible.")
if _tf_available:
    logging.info(f"TensorFlow Version: {tf.__version__}")
else:
    logging.info("TensorFlow no disponible.")

if _timezone_info_available and lima_tz:
    tz_provider = 'zoneinfo' if 'zoneinfo' in sys.modules else ('pytz' if 'pytz' in sys.modules else 'unknown')
    logging.info(f"Timezone: {str(lima_tz)} (Provider: {tz_provider})")
else:
    logging.warning("Librería de Timezone (zoneinfo/pytz) no disponible o zona inválida. Se usarán timestamps naive.")

# --- Fin Parte 1/13 ---
# -*- coding: utf-8 -*-
# =======================================================================================
# PARTE 2/13: TF GPU Check, Warnings Silencing, Initial Message (vD+DX v4 - CORREGIDO 25-Abr)
# =======================================================================================
# (Continuación desde Parte 1)

# --- TF GPU Check ---
logging.info("\n" + "="*30 + "\n--- CHECK TENSORFLOW GPU ---" + "\n" + "="*30)
if _tf_available:
    try:
        # Listar dispositivos físicos GPU detectados por TensorFlow
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Si se encontraron GPUs, loguear información
            logging.info(f"TF detectó {len(gpus)} GPU(s): {gpus}")
            # Intentar configurar memory growth para cada GPU
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logging.info(f"  Memory growth habilitado OK para {gpu.name}")
                except RuntimeError as e_mem:
                    # Advertir si memory growth no se puede setear
                    logging.warning(f"  WARN habilitando memory growth para {gpu.name}: {e_mem} (¿TF ya inicializado?)")
                except Exception as e_other_mem:
                    # Capturar otros posibles errores
                    logging.warning(f"  WARN inesperado habilitando memory growth para {gpu.name}: {e_other_mem}")
        else:
            # Si no se detectan GPUs, mostrar advertencia clara en consola y log
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! ADVERTENCIA CRÍTICA: TENSORFLOW NO DETECTÓ GPU !!!")
            print("!!! La predicción usará CPU (puede ser lenta).      !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.critical("\n" + "!"*10 + " ADVERTENCIA CRÍTICA: TF NO detectó GPU " + "!"*10 + "\n")
    except Exception as e_gpu_check:
        # Capturar cualquier error durante el proceso de chequeo de GPU
        logging.error(f"Error crítico durante chequeo de TF GPU: {e_gpu_check}", exc_info=True)
else:
    # Si TF no está disponible, loguear advertencia
    logging.warning("TensorFlow no disponible, omitiendo chequeo GPU.")
logging.info("--- FIN CHECK TENSORFLOW GPU ---")


# Silenciar logs y warnings específicos de librerías externas si son muy molestos
# Silenciar logs de Matplotlib (si se usara para plots)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
# Ignorar warnings de Pandas/Numpy sobre cambios futuros que no afectan el código actual
warnings.filterwarnings("ignore", category=FutureWarning)
# Ignorar ciertos UserWarning de Pandas que pueden aparecer con algunas operaciones
warnings.filterwarnings("ignore", category=UserWarning, module='pandas')


# --- Mensaje Inicial del Script ---
logging.info("="*60)
logging.info("--- INICIO SCRIPT BACKTEST/OPTIMIZACION (vD+DX v5 - runonce=False) ---") # Mensaje actualizado
logging.info(f"Logging Config: Nivel={logging.getLevelName(log_level)}, Archivo='{log_filename}'")
# Imprimir separador en consola para mayor claridad
print("-" * 60)

# --- Fin Parte 2/13 ---
# -*- coding: utf-8 -*-
# =======================================================================================
# PARTE 3/13: Función Auxiliar `calculate_grid_levels` (vD+DX v4 - CORREGIDO 25-Abr)
# =======================================================================================
# (Continuación desde Parte 2)

# =======================================================================================
# Función Auxiliar `calculate_grid_levels`
# =======================================================================================
# NOTA: Esta función está FUERA de la clase Strategy.
def calculate_grid_levels(upper_bound, lower_bound, num_grids, mode='LINEAR', price_precision=8):
    """
    Calcula los precios de los niveles del grid entre lower y upper bound.
    Maneja modos LINEAR/LOGARITHMIC, validaciones y correcciones.

    Args:
        upper_bound (float): Límite superior del grid.
        lower_bound (float): Límite inferior del grid.
        num_grids (int): Número de intervalos/espacios en el grid (niveles = num_grids + 1).
        mode (str): 'LINEAR' o 'LOGARITHMIC'.
        price_precision (int): Número de decimales para redondear los precios.

    Returns:
        list: Lista ordenada de precios de niveles, o lista vacía si hay error.
    """
    num_points = num_grids + 1 # Número total de puntos/niveles
    log_prefix = "[calculate_grid_levels]"

    # --- Validaciones iniciales de tipos y valores ---
    if not all(isinstance(arg, (int, float)) for arg in [upper_bound, lower_bound] if arg is not None) \
            or not isinstance(num_grids, int) or num_grids < 0:
        logging.error(f"{log_prefix} Argumentos inválidos: U={upper_bound}, L={lower_bound}, num_grids={num_grids}")
        return [] # Devolver lista vacía si los tipos son incorrectos

    if mode == 'LOGARITHMIC' and (lower_bound is None or lower_bound <= 1e-12): # Log necesita L > 0 estricto
        logging.error(f"{log_prefix} Límite inferior ({lower_bound}) <= 0 inválido para modo LOGARITHMIC.")
        return []
    if mode == 'LINEAR' and lower_bound is not None and lower_bound < 0: # Lineal puede tener L=0 pero no L<0
        logging.warning(f"{log_prefix} Límite inferior ({lower_bound}) < 0 para modo LINEAR. Usando 0.0.")
        lower_bound = 0.0
    if lower_bound is None or upper_bound is None: # Chequeo adicional por si acaso
        logging.error(f"{log_prefix} Límite inferior o superior es None. U={upper_bound}, L={lower_bound}")
        return []


    # Definir el "tick size" o mínima diferencia de precio basada en la precisión
    min_positive_price = 10**(-price_precision)
    # Separación mínima requerida entre niveles (la mitad de un tick para evitar solapamiento por redondeo)
    required_sep = min_positive_price / 2.0

    # --- Asegurar U > L (con una separación mínima) ---
    if upper_bound <= lower_bound + required_sep:
        # Si U no es suficientemente mayor que L, intentar ajustar U
        logging.warning(f"{log_prefix} U ({upper_bound}) <= L ({lower_bound}) + Sep ({required_sep}). Ajustando U.")
        print(f"ADVERTENCIA PRINT: {log_prefix} U ({upper_bound}) <= L ({lower_bound}) + Sep ({required_sep}). Intentando ajustar U.") # Mantener print si es útil
        # Ajustar U para que sea L + 1 tick mínimo
        upper_bound = lower_bound + min_positive_price
        # Re-check después del ajuste
        if upper_bound <= lower_bound + required_sep:
            # Si incluso después de añadir un tick siguen muy juntos, es un error
            logging.error(f"{log_prefix} Falló el ajuste U > L + Sep. U={upper_bound}, L={lower_bound}")
            return []
        logging.info(f"{log_prefix} U ajustado a {upper_bound:.{price_precision}f}")

    # --- Caso especial: num_grids = 0 (solo devuelve L y U) ---
    if num_grids == 0:
        logging.debug(f"{log_prefix} num_grids=0, devolviendo L y U redondeados.")
        try:
            l_round = round(lower_bound, price_precision)
            u_round = round(upper_bound, price_precision)
            # Asegurar que U redondeado siga siendo > L redondeado + separación mínima
            if u_round <= l_round + required_sep:
                logging.warning(f"{log_prefix} num_grids=0: U_round ({u_round}) <= L_round ({l_round}) + Sep post-redondeo. Ajustando U_round.")
                print(f"ADVERTENCIA PRINT: {log_prefix} num_grids=0: U_round ({u_round}) <= L_round ({l_round}) + Sep post-redondeo. Ajustando U_round.")
                # Ajustar U redondeado a L redondeado + 1 tick
                u_round = round(l_round + min_positive_price, price_precision)

            levels = sorted([l_round, u_round])
            # Verificar separación final después del ajuste
            if levels[0] >= levels[1] - required_sep:
                logging.error(f"{log_prefix} No se pudo separar L/U redondeados con num_grids=0. L={levels[0]}, U={levels[1]}")
                return []
            return levels # Devuelve lista con L y U redondeados y separados
        except Exception as e:
            logging.error(f"{log_prefix} Error redondeando L/U (num_grids=0): {e}", exc_info=True)
            return []

    # --- Cálculo de niveles base (Lineal o Logarítmico) ---
    levels = []
    try:
        if mode == 'LOGARITHMIC':
            # Asegurar que lower_bound sea > 0 para el logaritmo
            lower_bound_log = max(lower_bound, min_positive_price) # Usar al menos el tick size
             # Verificar separación de nuevo con L posiblemente ajustado
            if upper_bound <= lower_bound_log + required_sep:
                 logging.error(f"{log_prefix} U ({upper_bound}) no > L ajustado ({lower_bound_log}) + Sep para LOG.")
                 return []

            # Calcular logs y generar espacio logarítmico
            log_lower = np.log10(lower_bound_log)
            log_upper = np.log10(upper_bound)
            if not (np.isfinite(log_lower) and np.isfinite(log_upper)):
                 logging.error(f"{log_prefix} Logs inválidos (NaN/inf). L={lower_bound_log}, U={upper_bound}")
                 return []
            if log_lower >= log_upper:
                 logging.error(f"{log_prefix} log(L) >= log(U). L={lower_bound_log}, U={upper_bound}")
                 return []
            levels = np.logspace(log_lower, log_upper, num=num_points).tolist()
        else: # LINEAR (default)
            levels = np.linspace(float(lower_bound), float(upper_bound), num=num_points).tolist()
    except Exception as e:
        logging.error(f"{log_prefix} Error calculando niveles base ({mode}): {e}", exc_info=True)
        return []

    # --- Procesamiento post-cálculo (redondeo, unicidad, monotonía) ---
    if not levels:
        logging.error(f"{log_prefix} Cálculo base falló, lista niveles vacía.")
        return []

    try:
        # 1. Redondear y obtener únicos
        levels_rounded = [round(level, price_precision) for level in levels]
        levels_unique = sorted(list(set(levels_rounded)))

        # 2. Verificar si quedaron suficientes niveles únicos (>1)
        if len(levels_unique) < 2:
            logging.error(f"{log_prefix} < 2 niveles únicos post-redondeo/set. U={upper_bound}, L={lower_bound}, N={num_grids}")
            return []

        final_levels = levels_unique
        min_required_unique = max(2, int(num_points * 0.5)) # Ej: Al menos 2 y ~50% de los teóricos

        # 3. Chequear si U/L colapsaron o si hay muy pocos niveles -> forzar re-espaciado lineal
        needs_respacing = False
        if final_levels[0] >= final_levels[-1] - required_sep: # L >= U - epsilon
            logging.warning(f"{log_prefix} L ({final_levels[0]}) >= U ({final_levels[-1]}) - Sep post-redondeo/únicos. Forzando re-espaciado lineal.")
            print(f"ADVERTENCIA PRINT: {log_prefix} L>=U post-redondeo/únicos. Forzando re-espaciado lineal.")
            needs_respacing=True
        elif len(final_levels) < min_required_unique and num_grids > 1: # Muy pocos niveles únicos
            logging.warning(f"{log_prefix} Niveles únicos ({len(final_levels)}) < mín req ({min_required_unique}). Forzando re-espaciado lineal.")
            print(f"ADVERTENCIA PRINT: {log_prefix} Niveles únicos ({len(final_levels)}) < mín req ({min_required_unique}). Forzando re-espaciado lineal.")
            needs_respacing=True

        # 4. Realizar re-espaciado lineal si fue necesario
        if needs_respacing:
            logging.warning(f"{log_prefix} Re-espaciando linealmente entre L={lower_bound} y U={upper_bound} con {num_points} puntos...")
            start_lvl = lower_bound # Usar L/U originales
            # Asegurar un ancho mínimo total para el re-espaciado
            eff_grids = max(1, num_grids)
            min_width_needed = min_positive_price * eff_grids
            end_lvl = max(upper_bound, start_lvl + min_width_needed) # Asegurar U >= L + ancho mínimo

            final_levels = np.linspace(start_lvl, end_lvl, num=num_points).tolist()
            final_levels = [round(level, price_precision) for level in final_levels]
            final_levels = sorted(list(set(final_levels)))

            # Verificar resultado del re-espaciado
            if len(final_levels) < 2 or final_levels[0] >= final_levels[-1] - required_sep:
                logging.error(f"{log_prefix} Re-espaciado lineal forzado falló. Niveles finales: {final_levels}")
                return []
            logging.info(f"{log_prefix} Re-espaciado lineal forzado OK ({len(final_levels)} niveles únicos).")

        # 5. Verificar monotonía estricta usando Decimal para alta precisión
        tolerance = min_positive_price / 100.0 # Tolerancia muy pequeña
        is_mono = all((final_levels[i+1] - final_levels[i]) > tolerance for i in range(len(final_levels) - 1))

        if not is_mono:
            logging.warning(f"{log_prefix} Niveles no estrictamente monótonos. Intentando corrección con Decimal.")
            print(f"ADVERTENCIA PRINT: {log_prefix} Niveles no estrictamente monótonos. Intentando corrección con Decimal.")
            try:
                # Intenta corregir usando la librería Decimal
                Decimal = decimal.Decimal; context = decimal.Context(prec=price_precision + 5)
                quantizer = Decimal('1e-' + str(price_precision))
                unique_deci = sorted(list(set(context.quantize(Decimal(str(l)), quantizer) for l in final_levels)))
                levels_corr = [float(d) for d in unique_deci]

                # Verificar si la corrección funcionó
                if len(levels_corr) < 2 or not all((levels_corr[i+1] - levels_corr[i]) > tolerance for i in range(len(levels_corr) - 1)):
                    logging.error(f"{log_prefix} Corrección de monotonía con Decimal falló. Niveles: {levels_corr}")
                    return []
                logging.info(f"{log_prefix} Corrección de monotonía con Decimal OK ({len(levels_corr)} niveles).")
                final_levels = levels_corr # Usar los niveles corregidos
            except Exception as e_decimal:
                logging.error(f"{log_prefix} Error durante corrección con Decimal: {e_decimal}", exc_info=True)
                return []

        # 6. Advertencia final si se perdieron demasiados niveles (pero no falló)
        if len(final_levels) < num_points * 0.75 and num_grids > 1 and not needs_respacing:
             logging.warning(f"{log_prefix} Número final de niveles únicos ({len(final_levels)}) < 75% teóricos ({num_points}). Considerar revisar precisión o rangos U/L.")

    except Exception as e_postproc:
        logging.error(f"{log_prefix} Error inesperado post-procesando niveles: {e_postproc}", exc_info=True)
        return []

    # Éxito: Devolver la lista final de niveles
    logging.debug(f"{log_prefix} Cálculo niveles OK ({mode}, {len(final_levels)} niveles): {[f'{l:.{price_precision}f}' for l in final_levels]}")
    return final_levels

# --- Fin Parte 3/13 ---
# -*- coding: utf-8 -*-
# =======================================================================================
# PARTE 4/13: Indicador Personalizado BBWPercentIndicator (vD+DX v4 - CORREGIDO 25-Abr)
# =======================================================================================
# (Continuación desde Parte 3)

# ===========================================================
# Indicador Personalizado BBW%
# ===========================================================
class BBWPercentIndicator(bt.Indicator):
    """
    Calcula el Ancho de Banda de Bollinger (BBW) como porcentaje de la línea media.
    BBW% = ((BB_Top - BB_Bot) / BB_Mid) * 100

    Este indicador requiere que se le pase una instancia existente de BollingerBands
    a través del parámetro 'bb_range'.
    """
    # Nombre de la línea de salida que contendrá el BBW%
    lines = ('bbwp',)

    # Parámetros del indicador: solo necesita la referencia a las BB
    params = (
        ('bb_range', None), # Parámetro para pasar la instancia de BollingerBands
    )

    # Información para el trazado del gráfico (opcional)
    plotinfo = dict(subplot=True) # Dibujar en un panel separado

    # Configuración de cómo se verá la línea en el gráfico
    plotlines = dict(
        bbwp=dict(_name='BBW%', color='purple', ls='-.') # Nombre 'BBW%', color púrpura, línea dash-dot
    )

    def __init__(self):
        """Inicializador del indicador BBW%."""
        # Verificar que el parámetro 'bb_range' fue proporcionado y es válido
        if self.p.bb_range is None or not isinstance(self.p.bb_range, bt.indicators.BollingerBands):
             raise ValueError("BBWPercentIndicator requiere que se pase un indicador BollingerBands válido a través del parámetro 'bb_range'.")

        # Alias local para el indicador BollingerBands pasado
        bb = self.p.bb_range

        # Verificar que el objeto BB tiene las líneas necesarias (top, bot, mid)
        if not all(hasattr(bb.lines, line_name) for line_name in ['top', 'bot', 'mid']):
            raise ValueError("El objeto BollingerBands pasado en 'bb_range' no parece tener las líneas 'top', 'bot', 'mid'.")

        # Calcular el ancho de banda absoluto (Top - Bot)
        bbw = bb.lines.top - bb.lines.bot

        # Calcular el ratio porcentual respecto a la línea media (Mid)
        # Usar bt.DivByZero para manejar de forma segura casos donde bb.lines.mid sea 0 o muy cercano a 0
        bbw_ratio_safe = bt.DivByZero(bbw * 100.0, bb.lines.mid, zero=0.0)

        # Asignar el resultado a la línea de salida 'bbwp'
        # Se añade una comprobación adicional con bt.If para asegurar que mid > 0 (aunque DivByZero ya lo maneja)
        self.lines.bbwp = bt.If(bb.lines.mid > 1e-12, bbw_ratio_safe, 0.0)

        # Llamar al __init__ de la clase padre (bt.Indicator) es importante
        super(BBWPercentIndicator, self).__init__()

# --- Fin Parte 4/13 ---
# -*- coding: utf-8 -*-
# ============================================================================
# PARTE 5/13: Inicio Clase Estrategia, Params (vD+DX v4 - CORREGIDO 25-Abr)
# ============================================================================
# (Continuación desde Parte 4)

# ============================================================================
# Inicio Clase Estrategia, Params
# ============================================================================
class GridBacktraderStrategyV5(bt.Strategy):
    """
    Estrategia Grid Trading v5 (vD - SQLite + Diagnóstico Extendido v5 - runonce=False).
    - Implementa método next().
    - Implementa helper _get_order_amount().
    - notify_trade corregido para manejar trade=None y loguear trades.
    - Se ejecuta con runonce=False para fiabilidad de logging en notify_trade.
    """
    # ==========================================
    # === DEFINICIÓN DE PARÁMETROS (params) ===
    # ==========================================
    # Los parámetros permiten configurar la estrategia desde fuera,
    # ya sea con valores fijos o mediante optimización (Optuna).
    params = (
        # --- Parámetro para SQLite ---
        # Nombre del archivo de base de datos SQLite temporal usado por Optuna.
        # Se pasa desde objective_function durante la optimización IS. Es None en OOS.
        ('trial_db_name', None),

        # --- Configuración del Grid ---
        ('num_grids', 10),                     # Número de intervalos (niveles = num_grids + 1)
        ('grid_mode', 'LINEAR'),               # Modo de espaciado: 'LINEAR' o 'LOGARITHMIC'
        ('bound_change_threshold_percent', 15.0), # % cambio en Centro o Ancho para reiniciar grid

        # --- Tamaño de Órdenes ---
        ('dynamic_order_size_mode', 'PERCENT'),# 'PERCENT' (usa %) o 'FIXED' (usa USD)
        ('order_size_percent', 1.0),           # % del equity total por orden de nivel
        ('order_size_usd', 10.0),              # Tamaño fijo en USD por orden (si mode='FIXED')

        # --- Gestión de Riesgo Global ---
        ('stop_loss_percent', 2.0),            # % fuera de U/L activos para ejecutar SL global
        ('take_profit_percent', 10.0),         # % de ganancia sobre capital inicial para ejecutar TP global

        # --- Filtro de Rango (Condiciones Base para Activar Grid) ---
        ('adx_period', 14),                    # Periodo ADX
        ('adx_threshold', 25),                 # ADX debe estar POR DEBAJO para considerar rango
        ('bb_period_range', 20),               # Periodo BB para filtro BBW%
        ('bb_stddev_range', 2.0),              # Desv. Estándar BB para filtro BBW%
        ('bbw_threshold_percent', 3.0),        # BBW% debe estar POR DEBAJO para considerar rango

        # --- Determinación de Límites del Grid (U/L) ---
        ('bounds_method', 'ATR'),              # Método para calcular U/L: 'ATR' o 'BB'
        ('atr_period', 14),                    # Periodo ATR (si bounds_method='ATR')
        ('atr_multiplier', 1.8),               # Multiplicador ATR para ancho total del grid
        ('bb_period_bounds', 20),              # Periodo BB para límites (si bounds_method='BB')
        ('bb_stddev_bounds', 2.0),             # Desv. Estándar BB para límites (si bounds_method='BB')
        ('bb_multiplier_bounds', 1.5),         # Multiplicador del ancho BB para ancho total grid

        # --- Re-evaluación Periódica del Grid ---
        ('grid_update_interval', 96),          # Barras para re-evaluar límites (0=desactivado)

        # --- Activación de Filtros Opcionales ---
        # Estos parámetros solo indican si el filtro *podría* usarse.
        # La activación real depende de `strategy_complexity_level`.
        ('use_trend_filter', False),
        ('use_volume_filter', False),
        ('use_prediction_filter', False),

        # --- Parámetros Filtro de Tendencia ---
        ('trend_filter_ma_period', 100),       # Periodo SMA para tendencia
        ('trend_filter_max_deviation', 2.0),   # Máx. desviación % permitida del precio vs SMA

        # --- Parámetros Filtro de Volumen ---
        ('volume_filter_lookback', 20),        # Periodo SMA para volumen promedio
        ('volume_filter_min_mult', 0.8),       # Volumen actual debe ser >= AvgVol * mult

        # --- Parámetros Filtro de Predicción (Comunes) ---
        ('prediction_lookback', 100),          # Ventana datos para función de predicción

        # --- Parámetros Filtro de Predicción (Específicos - EJEMPLO TF) ---
        # (Estos dependen de la implementación en tu_funcion_de_prediccion)
        ('pred_roc_period', 5),
        ('pred_roc_threshold_pct', 1.0),       # Umbral ROC (abs) debe ser MENOR
        ('pred_vol_short_period', 5),
        ('pred_vol_long_period', 20),
        ('pred_vol_ratio_max', 2.0),           # Ratio Vol (corta/larga) debe ser MENOR
        ('pred_vol_ratio_min', 0.5),           # Ratio Vol (corta/larga) debe ser MAYOR
        ('pred_mrev_sma_period', 20),
        ('pred_mrev_std_period', 20),
        ('pred_mrev_zscore_threshold', 1.5),   # Umbral Z-Score (abs) debe ser MENOR

        # --- Control de Complejidad (Activa/Desactiva Filtros en Bloque) ---
        # Controla qué filtros se usan realmente en `_determine_active_filters`
        ('strategy_complexity_level', 'full'), # Opciones: 'full', 'no_prediction', 'no_pred_no_volume', 'no_filters'

        # --- Configuración de Reposicionamiento ---
        ('reposition_dist_ticks', 2),          # Distancia mín. (ticks) para colocar orden repo
        ('reposition_timeout_bars', 5),        # Barras máx. para esperar antes de descartar repo

        # --- Configuración de Precisión Numérica ---
        ('price_precision', 8),                # Decimales para precios
        ('amount_precision', 8),               # Decimales para cantidades/tamaños
    )
    # --- Fin del bloque params ---

    # --- Fin Parte 5/13 ---
# -*- coding: utf-8 -*-
# ============================================================================
# PARTE 6/13: Método __init__ de la Estrategia (vD+DX v4 - CORREGIDO 25-Abr)
# ============================================================================
# (Continuacion de la clase GridBacktraderStrategyV5 desde Parte 5)

    def __init__(self):
        """
        Inicializador de la estrategia.
        Configura datafeeds, logger, indicadores, variables de estado,
        calcula lookback mínimo y prepara la gestión de la DB SQLite.
        """
        # Llamar al inicializador de la clase base (bt.Strategy)
        super().__init__()

        # Configurar logger específico para esta instancia
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__qualname__}")
        self.log.info(f"Inicializando instancia estrategia (PID: {os.getpid()})...")

        # --- Validación y Asignación de Datos ---
        if not self.datas:
            self.log.error("¡Error Crítico! No se proporcionó data feed.")
            raise ValueError("Data feed requerido.")
        if len(self.datas) > 1:
            self.log.warning(f"Múltiples data feeds ({len(self.datas)}). Usando solo el primero ('{getattr(self.datas[0], '_name', 'data0')}').")

        self.data_main = self.datas[0] # Usar el primer data feed

        # Crear alias para acceso más corto a las líneas de datos
        try:
            self.data_open = self.data_main.open
            self.data_high = self.data_main.high
            self.data_low = self.data_main.low
            self.data_close = self.data_main.close
            self.data_volume = self.data_main.volume
            self.data_datetime = self.data_main.datetime
            # Test rápido para asegurar que el data feed es usable
            _ = len(self.data_close)
            self.log.debug(f"Data feed '{getattr(self.data_main, '_name', 'data0')}' OK.")
        except AttributeError as e:
            self.log.error(f"Data feed '{getattr(self.data_main, '_name', 'data0')}' inválido: falta {e}", exc_info=True)
            raise ValueError(f"Data feed inválido: falta {e}")
        except Exception as e_alias:
            self.log.error(f"Error creando alias de datos: {e_alias}", exc_info=True)
            raise RuntimeError(f"Fallo alias datos: {e_alias}")

        # --- Helper de Datetime ---
        # Intenta obtener datetime de la forma preferida y más rápida
        try:
            self.dt = lambda: self.data_main.datetime.datetime(0)
            _ = self.dt() # Test rápido
            self.log.debug("Usando self.dt via datetime.datetime(0).")
        except Exception:
            # Fallback si el método anterior falla
            try:
                self.dt = lambda: bt.num2date(self.data_datetime[0])
                _ = self.dt() # Test rápido
                self.log.warning("Usando fallback self.dt: bt.num2date(datetime[0]).")
            except Exception as e_dt:
                # Fallback final si todo falla (menos preciso)
                self.log.error(f"Error crítico helper dt: {e_dt}. Usando fallback: número barra.", exc_info=True)
                self.dt = lambda: f"Barra {len(self)}"

        # --- Creación de Instancias de Indicadores ---
        self.log.debug("Creando instancias de indicadores...")
        try:
            # Indicadores estándar
            self.adx = bt.indicators.ADX(self.data_main, period=self.p.adx_period)
            self.atr = bt.indicators.ATR(self.data_main, period=self.p.atr_period)
            self.sma_trend = bt.indicators.SimpleMovingAverage(self.data_close, period=self.p.trend_filter_ma_period)
            self.avg_volume = bt.indicators.SimpleMovingAverage(self.data_volume, period=self.p.volume_filter_lookback)

            # Bollinger Bands para filtro de rango
            self.bb_range = bt.indicators.BollingerBands(self.data_close,
                                                         period=self.p.bb_period_range,
                                                         devfactor=self.p.bb_stddev_range)
            # Indicador BBW% personalizado (usa bb_range)
            self.bbw_percent_indicator = BBWPercentIndicator(bb_range=self.bb_range)
            self.bbw_percent = self.bbw_percent_indicator.lines.bbwp # Alias a la línea de salida

            # Bollinger Bands para determinar límites (si se usa ese método)
            self.bb_bounds = bt.indicators.BollingerBands(self.data_close,
                                                         period=self.p.bb_period_bounds,
                                                         devfactor=self.p.bb_stddev_bounds)

            self.log.debug("Indicadores creados OK.")
        except Exception as e_ind:
            self.log.exception(f"ERROR CRÍTICO creando indicadores: {e_ind}") # Loguea el traceback completo
            raise RuntimeError(f"Fallo creación indicadores: {e_ind}") from e_ind # Relanza con causa

        # --- Variables de Estado Internas ---
        self.log.debug("Inicializando variables de estado...")
        self.grid_active = False             # Flag: ¿Está el grid activo?
        self.active_upper_bound = None       # Límite superior del grid activo
        self.active_lower_bound = None       # Límite inferior del grid activo
        self.grid_levels = []                # Lista de precios de los niveles del grid activo
        self.pending_orders = {}             # Diccionario para órdenes pendientes {ref: data}
        self.pending_repositions = {}        # Diccionario para reposiciones pendientes {level_idx: data}
        self.grid_initial_capital_usd = 0.0  # Capital al activar el grid (para TP global)
        self.last_bounds_update_bar = -1     # Barra de la última actualización/re-evaluación de límites

        # Flags internos para filtros (se actualizan en _determine_active_filters)
        self._active_use_trend_filter = False
        self._active_use_volume_filter = False
        self._active_use_prediction_filter = False

        # --- Cálculo del Lookback Mínimo Necesario ---
        # Recolectar todos los períodos de los indicadores/filtros/predicción
        periods = [
            self.p.adx_period,
            self.p.bb_period_range,
            self.p.atr_period,
            self.p.bb_period_bounds,
            self.p.trend_filter_ma_period,
            self.p.volume_filter_lookback,
            self.p.prediction_lookback, # Para obtener datos históricos
            # Añadir períodos específicos de la función de predicción si los usa internamente
            # Ejemplo para la función TF de ejemplo:
            getattr(self.p, 'pred_roc_period', 0) + 1, # ROC necesita N+1 barras
            getattr(self.p, 'pred_vol_long_period', 0),
            getattr(self.p, 'pred_mrev_sma_period', 0),
            getattr(self.p, 'pred_mrev_std_period', 0),
        ]
        try:
            # Filtrar solo períodos válidos (números > 0)
            valid_p = [p for p in periods if isinstance(p, (int, float)) and p > 0]
            # El lookback necesario es el máximo de los períodos válidos + un pequeño margen (e.g., 2)
            self._min_lookback_needed = int(max(valid_p)) + 2 if valid_p else 100 # Default 100 si no hay períodos válidos
            self.log.info(f"Lookback mínimo calculado: {self._min_lookback_needed} barras")
        except Exception as e_lb:
            self.log.warning(f"Error calculando lookback: {e_lb}. Usando default: 100")
            self._min_lookback_needed = 100

        # --- Configuración Conexión SQLite ---
        self.conn = None # Inicializar como None
        self.trial_db_name = self.p.trial_db_name # Obtener de params
        if self.trial_db_name is None:
            # Normal en OOS o si no se usa Optuna
            self.log.debug("Param 'trial_db_name' es None. No se conectará a DB (Esperado en OOS).")
        else:
            # Durante optimización IS, se conectará en start()
            self.log.info(f"Nombre DB trial recibido: '{self.trial_db_name}'. Conexión se intentará en start().")

        # --- Log Parámetros Efectivos ---
        try:
            # Crear diccionario de parámetros excluyendo el nombre de la DB
            params_log = {p: getattr(self.p, p) for p in self.p._getkeys() if p != 'trial_db_name'}
            # Convertir a string JSON formateado
            params_str = json.dumps(params_log, indent=2, default=str) # default=str para manejar tipos no serializables
            self.log.info(f"Params efectivos estrategia (excluye db_name):\n{params_str}")
        except Exception as e_p:
            self.log.warning(f"Warn: No se pudieron loguear params efectivos: {e_p}")
        # ---> INICIO DEBUG STRATEGY __init__ <---
        print(f"--- DEBUG STRATEGY __init__ (PID: {os.getpid()}) ---")
        try:
            print(f"  ID de self.datas[0] (data_main): {id(self.data_main)}")
            print(f"  Nombre de self.data_main: {getattr(self.data_main, '_name', 'N/A')}")
            # Intentar obtener longitud; puede ser 0 si Backtrader aún no la cargó del todo
            print(f"  Longitud self.data_main (len) en __init__: {len(self.data_main)}")
            # Es posible que intentar acceder a self.data_main.datetime[0] aquí dé error o 0.0
            # print(f"  Primer dt numérico en __init__: {self.data_main.datetime[0]}")
        except Exception as e_init_debug:
            print(f"  Error obteniendo info debug en __init__: {e_init_debug}")
        print(f"--- FIN DEBUG STRATEGY __init__ ---")
        # ---> FIN DEBUG STRATEGY __init__ <---
        self.log.debug(f"Fin __init__ instancia {id(self)}.")
        db_name_in_init = getattr(self.p, 'trial_db_name', '!!!INIT: NO ENCONTRADO!!!')
        self.log.critical(f"[{self.__class__.__qualname__} __init__ FIN] PID:{os.getpid()} - self.p.trial_db_name = {repr(db_name_in_init)}")
        print(f"CRITICAL PRINT INIT - PID:{os.getpid()} - self.p.trial_db_name = {repr(db_name_in_init)}")
        # --- Fin __init__ ---

# --- Fin Parte 6/13 ---
# -*- coding: utf-8 -*-
# =========================================================================
# PARTE 7/13: Método notify_order (vD+DX v4 - CORREGIDO 25-Abr)
# =========================================================================
# (Continuacion de la clase GridBacktraderStrategyV5 desde Parte 6)

    # --- notify_order (Manejo de Órdenes y Reposicionamiento) ---
    def notify_order(self, order):
        """
        Gestiona notificaciones sobre cambios en el estado de las órdenes.
        - Ignora estados intermedios (Submitted, Accepted).
        - Si una orden de grid original se completa, registra una solicitud
          de reposicionamiento en `self.pending_repositions`.
        - Si una orden de reposicionamiento se completa, no hace nada más.
        - Limpia la orden de `self.pending_orders` al finalizar (Completed, Failed).
        """
        # Obtener timestamp para logs de forma segura
        try:
            dt_obj=self.dt();
            dt_str=dt_obj.strftime('%Y-%m-%d %H:%M:%S') if isinstance(dt_obj, datetime.datetime) else str(dt_obj)
        except Exception:
            dt_str = f"Barra {len(self)}"
        # Prefijo para logs relacionados con esta orden específica
        log_prefix = f"[{dt_str}] ORD({order.ref})"

        # Ignorar estados intermedios para no saturar logs
        if order.status in [order.Submitted, order.Accepted]:
            return

        # --- Orden Completada ---
        if order.status == order.Completed:
            # Extraer detalles de ejecución de forma segura
            exec_price=np.nan; exec_size=np.nan; exec_comm=np.nan; side='???'
            try:
                if order.executed:
                    exec_price=order.executed.price
                    exec_size=order.executed.size
                    exec_comm=order.executed.comm
                    side = 'BUY' if order.isbuy() else 'SELL'
                # Loguear detalles de la orden completada
                self.log.info(f"{log_prefix} Status: {order.getstatusname()} - {side} @ ${exec_price:.{self.p.price_precision}f}, Sz={exec_size:.{self.p.amount_precision}f}, Comm=${exec_comm:.4f}")
            except Exception as e:
                self.log.error(f"{log_prefix} Excepción detalles ejecución orden completada: {e}", exc_info=True)

            # Buscar datos de la orden en nuestro diccionario de pendientes
            order_data_check = self.pending_orders.get(order.ref)

            # Lógica de Reposicionamiento:
            # Solo preparar repo si:
            # 1. El grid está activo.
            # 2. La orden completada estaba en nuestro diccionario de pendientes.
            # 3. NO era ya una orden de reposicionamiento.
            if self.grid_active and order_data_check and not order_data_check.get('is_reposition_order', False):
                self.log.debug(f"{log_prefix} Orden grid ORIGINAL {order.ref} completada. Preparando solicitud de reposicionamiento...")
                try:
                    # Extraer datos necesarios de la orden original completada
                    level_idx = order_data_check['level_idx']
                    executed_side = order_data_check['side'] # 'buy' o 'sell' (minúsculas)
                    price_to_place = order_data_check['price'] # Precio original del nivel

                    # Determinar el lado opuesto para la orden de reposicionamiento
                    opposite_side = 'sell' if executed_side == 'buy' else 'buy'
                    self.log.debug(f"{log_prefix} Preparando solicitud repo {opposite_side.upper()} Lvl {level_idx} @ ${price_to_place:.{self.p.price_precision}f}")

                    # Calcular el tamaño para la orden de reposicionamiento
                    potential_size = self._get_order_amount(price_to_place) # Usa el helper
                    self.log.debug(f"{log_prefix} Tamaño potencial repo calculado: {potential_size:.{self.p.amount_precision}f}")

                    min_order_size = 10**(-self.p.amount_precision)
                    # Solo registrar si el tamaño es válido
                    if potential_size >= min_order_size:
                        # Advertir si ya existía una solicitud para este nivel (no debería pasar si la lógica es correcta)
                        if level_idx in self.pending_repositions:
                            self.log.warning(f"{log_prefix} ¡Ya existe solicitud repo pendiente Lvl {level_idx}! Sobrescribiendo.")
                            print(f"ADVERTENCIA PRINT: {log_prefix} ¡Ya existe solicitud repo pendiente Lvl {level_idx}! Sobrescribiendo.")

                        # Añadir la solicitud al diccionario de reposiciones pendientes
                        self.pending_repositions[level_idx] = {
                            'side_to_place': opposite_side,
                            'price': price_to_place,
                            'potential_size': potential_size,
                            'bar_executed': len(self) # Barra en la que se completó la orden original
                        }
                        self.log.info(f"{log_prefix} Solicitud repo {opposite_side.upper()} Lvl {level_idx} registrada (Pendiente chequeo dist/timeout en next).")
                    else:
                        # Si el tamaño calculado es muy pequeño, no registrar la solicitud
                        self.log.warning(f"{log_prefix} Tamaño potencial repo ({potential_size:.{self.p.amount_precision}f}) < mínimo ({min_order_size}). No se registra solicitud repo.")

                except KeyError as e_key:
                    # Error si falta alguna clave esperada en order_data_check
                    self.log.error(f"{log_prefix} EXCEPCIÓN KeyError preparando repo para Lvl {order_data_check.get('level_idx', '??')}: Falta clave {e_key}?", exc_info=True)
                except Exception as e_repo:
                    # Capturar cualquier otro error inesperado durante la preparación del repo
                    self.log.error(f"{log_prefix} EXCEPCIÓN inesperada preparando repo Lvl {order_data_check.get('level_idx', '??')}: {e_repo}", exc_info=True)

            # Si la orden completada estaba en pendientes pero era de repo O el grid estaba inactivo
            elif order.ref in self.pending_orders:
                is_repo_flag = self.pending_orders[order.ref].get('is_reposition_order', False)
                self.log.debug(f"{log_prefix} Orden {order.ref} completada. {'Era repo.' if is_repo_flag else ''} {'Grid inactivo.' if not self.grid_active else ''} No se prepara nuevo repo.")

            # Si la orden completada NO estaba en pendientes (pudo ser cierre por SL/TP global o desactivación)
            elif self.grid_active:
                 self.log.debug(f"{log_prefix} Orden {order.ref} completada no en pendientes (¿Cierre global?). No repo.")
            else:
                 # Caso raro: grid inactivo y orden completada no estaba en pendientes
                 self.log.debug(f"{log_prefix} Orden {order.ref} completada con grid inactivo y no en pendientes.")

            # Limpiar la orden completada del diccionario de pendientes (si estaba)
            self._remove_order_from_pending(order.ref, log_prefix, "Completada") # Helper definido más adelante

        # --- Orden Fallida (Cancelada, Rechazada, Expirada, Margen) ---
        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            status_name = order.getstatusname()
            # Imprimir y loguear como advertencia/error
            self.log.warning(f"{log_prefix} Status: {status_name} (Orden Fallida/Rechazada)")
            print(f"ERROR PRINT: {log_prefix} Status: {status_name} (Orden Fallida/Rechazada)") # Mantener print si es útil

            # Verificar si la orden fallida estaba en nuestro registro
            order_data = self.pending_orders.get(order.ref)
            if order_data:
                level_idx = order_data.get('level_idx')
                # Si falla una orden original (no repo) y había una solicitud de repo pendiente para ese nivel, advertir
                if level_idx is not None and not order_data.get('is_reposition_order', False) and level_idx in self.pending_repositions:
                     self.log.warning(f"{log_prefix} Orden original Lvl {level_idx} falló ({status_name}). Solicitud repo pendiente asociada podría no activarse.")

            # Limpiar la orden fallida del diccionario de pendientes (si estaba)
            self._remove_order_from_pending(order.ref, log_prefix, f"Fallida ({status_name})")

        else:
            # Otros estados posibles (Partial, etc.) - loguear como debug si es necesario
            self.log.debug(f"{log_prefix} Status intermedio/inesperado: {order.getstatusname()}")

    # --- Fin notify_order ---

# --- Fin Parte 7/13 ---
# -*- coding: utf-8 -*-
# =========================================================================
# PARTE 8/13: Método notify_trade (vD+DX v5 - runonce=False)
# =========================================================================
# (Continuacion de la clase GridBacktraderStrategyV5 desde Parte 7)

    # --- notify_trade (Versión corregida para runonce=False) ---
    def notify_trade(self, trade):
        """
        Gestiona notificaciones sobre trades (creación, apertura, cierre).
        MODIFICADO: Guarda pnlcomm y barlen de trades CERRADOS en DB SQLite externa.
        Incluye manejo de 'trade is None' y logs TRACE activos.
        Diseñado para funcionar con runonce=False.
        """
        # Logs iniciales para diagnosticar qué objeto llega
        self.log.critical(f"!!! INSIDE notify_trade: type(trade)={type(trade)}, trade={repr(trade)}")
        print(f"!!! PRINT INSIDE notify_trade: type(trade)={type(trade)}, trade={repr(trade)}")

        # Obtener timestamp y prefijo de forma segura
        try:
            dt_obj=self.dt();
            dt_str=dt_obj.strftime('%Y-%m-%d %H:%M:%S') if isinstance(dt_obj, datetime.datetime) else str(dt_obj)
        except Exception:
            dt_str = f"Barra {len(self)}"
        log_prefix = f"[{dt_str}] TRD({getattr(trade, 'ref', 'N/A')})"
        
        self.log.debug(f"{log_prefix} Check Conn al inicio de notify_trade: self.conn = {repr(self.conn)}")

        # Manejo explícito del caso 'trade is None'
        if trade is None:
            self.log.error(f"{log_prefix} Received trade object is None from Backtrader. Cannot process this notification.")
            print(f"ERROR PRINT: {log_prefix} Received trade object is None.")
            return # Salir si es None

        # Si llegamos aquí, 'trade' NO es None.
        self.log.critical(f"!!!!!!!! notify_trade LLAMADO (VALIDO)! trade_ref={trade.ref}, status={trade.status} !!!!!!!!!!")
        print(f"!!!!!!!! notify_trade LLAMADO (VALIDO)! trade_ref={trade.ref}, status={trade.status} !!!!!!!!!!")

        # Procesar basado en el estado del trade
        try:
            # --- Lógica para Trade CERRADO ---
            if trade.status == 2:
                self.log.info(f"{log_prefix} Procesando trade CERRADO (Ref: {trade.ref})...")

                # Inicializar variables locales
                pnl=np.nan; comm=np.nan; pnlcomm=np.nan; open_price_avg=np.nan
                size=np.nan; close_price_avg=np.nan; barlen=0; open_size_total = 0

                try:
                    # Intentar extraer todos los datos necesarios
                    open_price_avg = trade.price
                    size = trade.size
                    pnl = trade.pnl
                    comm = trade.commission
                    pnlcomm = trade.pnlcomm # Leer PnL Neto
                    barlen = trade.barlen   # Leer Duración

                    # Imprimir TRACE logs (ACTIVOS)
                    print(f"TRACE {log_prefix}: open_price_avg={open_price_avg}")
                    print(f"TRACE {log_prefix}: size={size}")
                    print(f"TRACE {log_prefix}: pnl={pnl}")
                    print(f"TRACE {log_prefix}: comm={comm}")
                    print(f"TRACE {log_prefix}: pnlcomm={pnlcomm}")
                    print(f"TRACE {log_prefix}: barlen={barlen}")

                    # Calcular precio cierre promedio (para logging)
                    if trade.historyon and trade.history:
                        for event in trade.history:
                            if event.status in [1, 3]: # OPEN o INCREASED
                                event_size = getattr(event.event, 'size', 0)
                                if event_size:
                                    open_size_total += abs(event_size)

                        if abs(open_size_total) > 10**(-self.p.amount_precision - 1) and np.isfinite(pnl):
                            close_price_avg = open_price_avg + (pnl / open_size_total)
                        else:
                            self.log.warning(f"{log_prefix} Tamaño apertura ({open_size_total:.{self.p.amount_precision}f}) no determinado o PnL inválido ({pnl}). ClosePrice=NaN.")
                    else:
                         self.log.warning(f"{log_prefix} Historial de trade no disponible/activo. ClosePrice=NaN.")
                    print(f"TRACE {log_prefix}: close_price_avg={close_price_avg}") # Imprimir aunque sea NaN

                    # Loguear detalles formateados
                    pnlcomm_fmt = f"{pnlcomm:.2f}" if np.isfinite(pnlcomm) else "NaN"
                    close_price_fmt = f"{close_price_avg:.{self.p.price_precision}f}" if np.isfinite(close_price_avg) else "NaN"
                    self.log.info(f"{log_prefix} TRADE CERRADO: Ref={trade.ref}, "
                                  f"Sz={size:.{self.p.amount_precision}f}, Open@~${open_price_avg:.{self.p.price_precision}f}, "
                                  f"Close@~${close_price_fmt}, PnL={pnl:.2f}, Comm={comm:.2f}, "
                                  f"Net={pnlcomm_fmt}, Bars={barlen}")

                    # Log antes de chequear condición DB
                    self.log.debug(f"{log_prefix} DB Check Vals: pnlcomm={pnlcomm}, barlen={barlen}, isfinite={np.isfinite(pnlcomm)}")

                    # --- Bloque para Guardar en SQLite ---
                    if self.conn and pnlcomm is not None and np.isfinite(pnlcomm) and barlen is not None:
                        self.log.debug(f"{log_prefix} Condición DB Write OK. Intentando INSERT con PnL={pnlcomm}, Bars={barlen}...")
                        try:
                            cursor = self.conn.cursor()
                            # >>> ¡¡INSERT ACTIVADO!! <<<
                            cursor.execute("INSERT INTO trades (pnlcomm, barlen) VALUES (?, ?)", (pnlcomm, barlen))
                            self.conn.commit()
                            self.log.info(f"{log_prefix} ¡ÉXITO! Trade PnL={pnlcomm:.2f}, Bars={barlen} GUARDADO en DB {self.trial_db_name}")
                            print(f"INFO PRINT: {log_prefix} Trade GUARDADO en DB.")
                        except sqlite3.Error as e_db:
                            self.log.error(f"{log_prefix} ¡ERROR SQLITE ESPECIFICO al guardar! PnL={pnlcomm}, Bars={barlen}. Error: {e_db}", exc_info=True)
                            print(f"ERROR PRINT: {log_prefix} ¡ERROR SQLITE ESPECIFICO!: {e_db}")
                        except Exception as e_db_other:
                            self.log.error(f"{log_prefix} ¡ERROR INESPERADO al guardar en SQLite! PnL={pnlcomm}, Bars={barlen}. Error: {e_db_other}", exc_info=True)
                            print(f"ERROR PRINT: {log_prefix} ¡ERROR INESPERADO DB WRITE!: {e_db_other}")
                    else:
                        # Si la condición falló (inesperado ahora, pero por si acaso)
                        self.log.warning(f"{log_prefix} Condición escritura DB falló (INESPERADO?): PnLNeto ({pnlcomm}) o Barlen ({barlen}) inválido? Conn={self.conn is not None}. Trade NO guardado.")
                        print(f"WARN PRINT: {log_prefix} Trade NO guardado en DB (condición if falló - INESPERADO?).")

                except Exception as e_closed:
                    # Captura error al procesar datos del trade (price, size, pnl, etc.)
                    # Usar print y traceback porque self.log.error fallaba aquí antes
                    print(f"!!!!!!!!!!!!!! ERROR EXCEPTION in notify_trade (status=2) !!!!!!!!!!!!!!")
                    print(f"ERROR Type: {type(e_closed).__name__}")
                    print(f"ERROR Args: {e_closed.args}")
                    import traceback
                    traceback.print_exc() # Directo a stderr
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # self.log.error(f"{log_prefix} EXCEPCION CAPTURADA procesando trade CERRADO (Ref {trade.ref}): {e_closed}", exc_info=True) # Comentado temporalmente

            # --- Lógica para Trade ABIERTO ---
            elif trade.status == 1:
                self.log.debug(f"{log_prefix} Trade ABIERTO/Actualizado: Ref={trade.ref}, Status={trade.status}, Sz={trade.size:.{self.p.amount_precision}f}")

            # --- Lógica para Trade CREADO ---
            elif trade.status == 0:
                self.log.debug(f"{log_prefix} Trade CREADO: Ref={trade.ref}")

        except Exception as e_trade_status:
            # Error Gral al acceder a trade.ref o trade.status (no debería pasar si trade no es None)
            self.log.error(f"{log_prefix} EXCEPCION Gral procesando trade (Ref {getattr(trade, 'ref', 'N/A')}): {e_trade_status}", exc_info=True)

    # --- Fin notify_trade ---

# --- Fin Parte 8/13 ---
# -*- coding: utf-8 -*-
# =========================================================================
# PARTE 9/13: Métodos start, stop, _determine_active_filters (vD+DX v5 - runonce=False)
# =========================================================================
# (Continuacion de la clase GridBacktraderStrategyV5 desde Parte 8)

    # --- Método start() --- (Abre conexión DB si aplica) ---
    def start(self):
        log_prefix = f"[{self.__class__.__qualname__} Start|PID:{os.getpid()}]"
        # ---> INICIO DEBUG STRATEGY start() <---
        print(f"\n--- DEBUG STRATEGY start() (PID: {os.getpid()}) ---")
        try:
            data_name = getattr(self.data_main, '_name', 'N/A')
            # En start(), los datos YA deberían estar cargados por Backtrader
            start_dt_num = self.data_main.datetime[0]
            end_dt_num = self.data_main.datetime[-1]
            start_date_str = bt.num2date(start_dt_num).isoformat() if not np.isnan(start_dt_num) else "NaN"
            end_date_str = bt.num2date(end_dt_num).isoformat() if not np.isnan(end_dt_num) else "NaN"
            num_bars = len(self.data_main)

            print(f"  ID de self.data_main en start(): {id(self.data_main)}")
            print(f"  Nombre de self.data_main: {data_name}")
            print(f"  Num Barras cargadas en self.data_main: {num_bars}")
            print(f"  Fecha Inicio (num/convertida): {start_dt_num} / {start_date_str}")
            print(f"  Fecha Fin (num/convertida): {end_dt_num} / {end_date_str}")
        except Exception as e_start_debug:
            print(f"  Error obteniendo info debug en start: {e_start_debug}")
        print(f"--- FIN DEBUG STRATEGY start() ---\n")
        # ---> FIN DEBUG STRATEGY start() <---
        db_name_param = getattr(self.p, 'trial_db_name', '!!!ATRIBUTO_NO_ENCONTRADO!!!')
        self.log.critical(f"{log_prefix} INICIO START - self.p.trial_db_name = {repr(db_name_param)}")
        print(f"CRITICAL PRINT START - self.p.trial_db_name = {repr(db_name_param)}")

        self.conn = None # Inicializar a None

        if db_name_param and db_name_param != '!!!ATRIBUTO_NO_ENCONTRADO!!!':
            self.log.info(f"{log_prefix} 'trial_db_name' OK. Intentando conectar a: {db_name_param}") # Cambiado a INFO
            print(f"INFO PRINT START - Intentando conectar a: {db_name_param}") # Cambiado a INFO

            # ---> RESTAURAR TRY/EXCEPT <---
            try:
                self.conn = sqlite3.connect(db_name_param, check_same_thread=False, timeout=15.0)
                self.log.info(f"{log_prefix} sqlite3.connect LLAMADO. Objeto conn: {repr(self.conn)}") # Cambiado a INFO
                print(f"INFO PRINT START - sqlite3.connect LLAMADO. Objeto conn: {repr(self.conn)}")

                # Verificar si la conexión funcionó y si la tabla existe
                if self.conn:
                    cursor = self.conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades';")
                    table_exists = cursor.fetchone()
                    if table_exists:
                        self.log.info(f"{log_prefix} Verificación post-conexión OK: Tabla 'trades' encontrada.") # Cambiado a INFO
                        print(f"INFO PRINT START - Verificación post-conexión OK.")
                    else:
                        # Esto sería un error grave, la tabla DEBE existir
                        self.log.error(f"{log_prefix} ¡ERROR CRÍTICO! Conexión establecida pero tabla 'trades' NO encontrada en {db_name_param}.")
                        print(f"ERROR PRINT START - ¡Tabla 'trades' NO encontrada!")
                        self.conn.close() # Intentar cerrar
                        self.conn = None # Marcar como fallido
                else:
                    # Si sqlite3.connect devolvió None
                    self.log.error(f"{log_prefix} ¡ERROR! sqlite3.connect devolvió None para {db_name_param}.")
                    print(f"ERROR PRINT START - sqlite3.connect devolvió None.")
                    # self.conn ya es None

            except sqlite3.Error as e_db_connect:
                self.log.error(f"{log_prefix} ¡ERROR SQLITE al conectar/verificar DB {db_name_param}! Error: {e_db_connect}", exc_info=True)
                print(f"ERROR PRINT START - ¡ERROR SQLITE!: {e_db_connect}")
                if self.conn: # Si hubo conexión parcial, intentar cerrar
                    try: self.conn.close()
                    except: pass
                self.conn = None # Asegurar que conn es None después del error

            except Exception as e_connect_other:
                self.log.error(f"{log_prefix} ¡ERROR INESPERADO al conectar/verificar DB {db_name_param}! Error: {e_connect_other}", exc_info=True)
                print(f"ERROR PRINT START - ¡ERROR INESPERADO CONEXIÓN!: {e_connect_other}")
                if self.conn: # Si hubo conexión parcial, intentar cerrar
                    try: self.conn.close()
                    except: pass
                self.conn = None # Asegurar que conn es None después del error
            # ---> FIN RESTAURAR TRY/EXCEPT <---

        else:
            self.log.warning(f"{log_prefix} 'trial_db_name' es None o inválido. No se conecta (esperado en OOS).") # Cambiado a WARNING
            print(f"WARN PRINT START - 'trial_db_name' es None. No se conecta (OOS).") # Cambiado a WARN

        # Log final del estado de self.conn DENTRO de start()
        self.log.info(f"{log_prefix} Fin start(). Estado final self.conn: {repr(self.conn)}")
        print(f"INFO PRINT START - Fin start(). Estado final self.conn: {repr(self.conn)}")


    # --- Método stop() --- (Cierra conexión DB si aplica, loguea estado final) ---
    def stop(self):
        """ Llamado al final. Cierra conexión DB y loguea estado final del portfolio y pendientes. """
        log_prefix = f"[{self.__class__.__qualname__} Stop|PID:{os.getpid()}]"
        self.log.info(f"{log_prefix} Método stop() llamado.")
        try:
            # Loguear valor final del portfolio
            final_value = self.broker.getvalue()
            self.log.info(f"{log_prefix} Valor final del portfolio: ${final_value:,.2f}")
        except Exception as e:
            self.log.error(f"{log_prefix} Error obteniendo valor final en stop(): {e}")

        # Loguear posición final específicamente si estamos en la fase IS
        if self.trial_db_name: # Identifica si es una ejecución IS
            pos_size_final = self.position.size if self.position else 0.0
            self.log.info(f"{log_prefix} Posición final al terminar Backtest IS: {pos_size_final:.{self.p.amount_precision}f}")

        # Advertir si quedaron órdenes o reposiciones pendientes (útil para debug)
        if self.pending_orders:
            self.log.warning(f"{log_prefix} ¡ADVERTENCIA! {len(self.pending_orders)} órdenes en 'pending_orders' al finalizar:")
            print(f"ADVERTENCIA PRINT: {log_prefix} ¡{len(self.pending_orders)} órdenes pendientes al finalizar!") # Mantener print si es útil
            for i, (ref, data) in enumerate(list(self.pending_orders.items())):
                if i >= 5: self.log.warning(f"{log_prefix}      ... y {len(self.pending_orders)-i} más."); break
                # Intentar obtener info de la orden de forma segura
                order_obj = data.get('order')
                order_status = order_obj.getstatusname() if order_obj else 'N/A'
                order_info = f"Side:{data.get('side','?')}, P:${data.get('price','?'):.{self.p.price_precision}f}, St:{order_status}"
                info_extra = " (Repo)" if data.get('is_reposition_order') else (f", Lvl:{data.get('level_idx','?')}" if 'level_idx' in data else "")
                self.log.warning(f"{log_prefix}      - Ref {ref}: {order_info}{info_extra}")
        if self.pending_repositions:
             self.log.warning(f"{log_prefix} ¡ADVERTENCIA! {len(self.pending_repositions)} solicitudes repo en 'pending_repositions' al finalizar:")
             print(f"ADVERTENCIA PRINT: {log_prefix} ¡{len(self.pending_repositions)} reposiciones pendientes al finalizar!") # Mantener print si es útil
             for i, (level_idx, data) in enumerate(list(self.pending_repositions.items())):
                 if i >= 5: self.log.warning(f"{log_prefix}      ... y {len(self.pending_repositions)-i} más."); break
                 self.log.warning(f"{log_prefix}      - Lvl {level_idx}: Pendiente {data.get('side_to_place','?').upper()} @ ${data.get('price','?'):.{self.p.price_precision}f} (desde B{data.get('bar_executed','?')})")

        # Cerrar conexión SQLite si estaba abierta
        self.log.debug(f"{log_prefix} Intentando cerrar conexión SQLite a '{self.trial_db_name}'...")
        if self.conn:
            try:
                self.conn.close()
                self.log.info(f"{log_prefix} Conexión SQLite cerrada OK.")
                self.conn = None # Marcar como cerrada
            except sqlite3.Error as e_db_close:
                self.log.error(f"{log_prefix} ERROR cerrando conexión SQLite: {e_db_close}", exc_info=True)
            except Exception as e_close_other:
                 self.log.error(f"{log_prefix} ERROR inesperado cerrando DB: {e_close_other}", exc_info=True)
        else:
            self.log.debug(f"{log_prefix} Conexión SQLite ya estaba cerrada o nunca se abrió.")

        self.log.info(f"{log_prefix} Fin del método stop().")
    # --- FIN MÉTODO stop() ---


    # --- Helper para determinar filtros activos ---
    def _determine_active_filters(self):
        """
        Actualiza los flags internos _active_use_* basado en el parámetro
        strategy_complexity_level y los parámetros individuales use_*.
        """
        level = getattr(self.p, 'strategy_complexity_level', 'full') # Default 'full'
        log_prefix=f"[_determine_active_filters|Lvl:{level}]"
        self.log.debug(f"{log_prefix} Determinando filtros activos...")

        # Obtener los valores base de los parámetros use_*
        use_trend_param = getattr(self.p, 'use_trend_filter', False)
        use_volume_param = getattr(self.p, 'use_volume_filter', False)
        use_predict_param = getattr(self.p, 'use_prediction_filter', False)

        # Aplicar lógica según el nivel de complejidad
        if level == 'full':
            # Usar lo que digan los parámetros individuales
            self._active_use_trend_filter = use_trend_param
            self._active_use_volume_filter = use_volume_param
            self._active_use_prediction_filter = use_predict_param
        elif level == 'no_prediction':
            # Usar trend y volume según params, pero forzar predicción a False
            self._active_use_trend_filter = use_trend_param
            self._active_use_volume_filter = use_volume_param
            self._active_use_prediction_filter = False
        elif level == 'no_pred_no_volume':
            # Usar trend según param, forzar volume y predicción a False
            self._active_use_trend_filter = use_trend_param
            self._active_use_volume_filter = False
            self._active_use_prediction_filter = False
        elif level == 'no_filters':
            # Forzar todos los filtros a False
            self._active_use_trend_filter = False
            self._active_use_volume_filter = False
            self._active_use_prediction_filter = False
        else:
            # Nivel no reconocido, usar 'full' como fallback seguro
            self.log.warning(f"{log_prefix} Nivel complejidad '{level}' no reconocido. Usando 'full'.")
            self._active_use_trend_filter = use_trend_param
            self._active_use_volume_filter = use_volume_param
            self._active_use_prediction_filter = use_predict_param

        # Loguear los filtros que se aplicarán *realmente* en esta activación
        self.log.info(f"{log_prefix} Filtros aplicables en esta activación: Trend={self._active_use_trend_filter}, Volume={self._active_use_volume_filter}, Predict={self._active_use_prediction_filter}")
    # --- Fin _determine_active_filters ---

# --- Fin Parte 9/13 ---
# -*- coding: utf-8 -*-
# =========================================================================
# PARTE 10/13: Método next (vD+DX v5 - runonce=False)
# =========================================================================
# (Continuacion de la clase GridBacktraderStrategyV5 desde Parte 9)

    def next(self):
        """
        Lógica principal ejecutada en cada barra por Backtrader.
        Sigue un orden estricto:
        1. Esperar fin de warm-up.
        2. Procesar reposiciones pendientes.
        3. Re-evaluar grid si está activo y toca por intervalo.
        4. Intentar activar grid si está inactivo y se cumplen condiciones.
        5. Chequear SL/TP globales si el grid está activo.
        """
        # Log Crítico para seguir el flujo principal de next
        self.log.critical(f"!!! NEXT BAR {len(self)} - Time: {self.dt()} - Grid Active: {self.grid_active} !!!")

        # --- 1. Warm-up Check ---
        if len(self) <= self._min_lookback_needed:
            # Durante el warm-up, no hacer nada más que calcular indicadores
            # self.log.debug(f"Bar {len(self)}: Warm-up ({self._min_lookback_needed} needed)")
            return

        # Log Crítico para indicar que el warm-up ha pasado
        self.log.critical(f"!!! NEXT BAR {len(self)} - PASÓ WARM-UP !!!")

        # Obtener timestamp seguro y prefijo log
        try:
            dt_obj=self.dt();
            dt_str=dt_obj.strftime('%Y-%m-%d %H:%M:%S') if isinstance(dt_obj, datetime.datetime) else str(dt_obj)
        except Exception:
            dt_str = f"Barra {len(self)}"
        log_prefix = f"[{self.__class__.__qualname__}|Next|{dt_str}]"
        
        # ---> INICIO DEBUG STRATEGY next() <---
        # Imprimir solo para las primeras 5 barras después del warm-up para no saturar
        if len(self) < self._min_lookback_needed + 5:
            try:
                current_dt_val = self.dt() # Llama a tu función helper dt()
                current_dt_num = self.data_main.datetime[0] # Acceso directo al número de BT
                current_dt_num_conv = bt.num2date(current_dt_num).isoformat() if not np.isnan(current_dt_num) else "NaN"

                print(f"--- DEBUG STRATEGY next() Bar:{len(self)} (PID:{os.getpid()}) ---")
                print(f"  Valor self.dt() reportado: {current_dt_val}")
                print(f"  Valor self.data_main.datetime[0] (num): {current_dt_num}")
                print(f"  Valor self.data_main.datetime[0] (conv): {current_dt_num_conv}")
                print(f"--- FIN DEBUG STRATEGY next() Bar:{len(self)} ---")
            except Exception as e_next_debug:
                print(f"--- ERROR DEBUG STRATEGY next() Bar:{len(self)}: {e_next_debug} ---")
        # ---> FIN DEBUG FECHAS EN NEXT <---

        # --- 2. Procesar Reposiciones Pendientes ---
        self.log.critical(f"!!! NEXT BAR {len(self)} - ANTES Procesar Repos ({len(self.pending_repositions)}) !!!")
        if self.pending_repositions:
            # Crear copia para iterar seguro si se modifican elementos
            repos_to_check = list(self.pending_repositions.items())
            current_price = self.data_close[0] # Necesario para check distancia

            for level_idx, repo_data in repos_to_check:
                self.log.debug(f"{log_prefix} Procesando repo pendiente Lvl {level_idx}...")
                price_to_place = repo_data['price']
                side_to_place = repo_data['side_to_place']
                bar_executed = repo_data['bar_executed']
                potential_size = repo_data['potential_size']

                # Check distancia (precio actual vs nivel donde se ejecutó orden original)
                min_dist_price = self.p.reposition_dist_ticks * (10**(-self.p.price_precision))
                price_diff = abs(current_price - price_to_place)

                # Check timeout (barras pasadas desde ejecución orden original)
                bars_passed = len(self) - bar_executed
                timeout_reached = bars_passed >= self.p.reposition_timeout_bars

                place_repo = False
                if price_diff >= min_dist_price:
                    self.log.debug(f"{log_prefix} Lvl {level_idx}: Distancia OK (Diff {price_diff:.{self.p.price_precision}f} >= {min_dist_price:.{self.p.price_precision}f}). Colocando repo.")
                    place_repo = True
                elif timeout_reached:
                    self.log.warning(f"{log_prefix} Lvl {level_idx}: Timeout ({bars_passed} >= {self.p.reposition_timeout_bars}) alcanzado para repo. DESCARTANDO.")
                    # Eliminar del diccionario si expira
                    try:
                        del self.pending_repositions[level_idx]
                    except KeyError:
                         pass # Ya podría haber sido procesado/eliminado
                    continue # Pasar al siguiente repo pendiente
                else:
                     # Aún no se cumplen condiciones de distancia ni timeout
                     self.log.debug(f"{log_prefix} Lvl {level_idx}: Distancia AÚN no cumplida (Diff {price_diff:.{self.p.price_precision}f} < {min_dist_price:.{self.p.price_precision}f}) y sin timeout. Esperando...")

                # Si se cumplen las condiciones, colocar la orden de reposición
                if place_repo:
                    try:
                        order_func = self.buy if side_to_place == 'buy' else self.sell
                        order = order_func(price=price_to_place, size=potential_size, exectype=bt.Order.Limit, data=self.data_main)

                        if order and order.ref:
                            # Registrar la orden de repo en pending_orders para seguimiento
                            self.pending_orders[order.ref] = {
                                'level_idx': level_idx, # Guardamos el nivel original
                                'side': side_to_place,
                                'price': price_to_place,
                                'size': potential_size,
                                'order': order,
                                'is_reposition_order': True # Marcar como orden de repo
                            }
                            self.log.info(f"{log_prefix} -> OK Repo {side_to_place.upper()} Lvl {level_idx} @ ${price_to_place:.{self.p.price_precision}f} (Sz={potential_size:.{self.p.amount_precision}f}) Ref:{order.ref}.")

                            # Eliminar del diccionario de pendientes una vez colocada la orden
                            try:
                                del self.pending_repositions[level_idx]
                            except KeyError:
                                self.log.warning(f"{log_prefix} Lvl {level_idx} ya no estaba en pending_repositions al intentar borrar.")
                        else:
                             # Error si Backtrader no devuelve una orden válida
                             self.log.error(f"{log_prefix} Fallo crear/reg orden REPO {side_to_place.upper()} Lvl {level_idx}. Orden devuelta: {order}")

                    except Exception as e_repo_place:
                         self.log.error(f"{log_prefix} EXCEPCION colocando orden REPO {side_to_place.upper()} Lvl {level_idx} @ ${price_to_place:.{self.p.price_precision}f}: {e_repo_place}", exc_info=True)
                         # Considerar eliminar de pending_repositions aquí también si falla
                         try:
                             del self.pending_repositions[level_idx]
                         except KeyError:
                             pass

        # --- 3. Re-evaluación Periódica (Si Grid Activo) ---
        self.log.critical(f"!!! NEXT BAR {len(self)} - ANTES Re-evaluar Grid (Activo: {self.grid_active}) !!!")
        # Solo re-evaluar si el grid está activo y el intervalo es > 0
        if self.grid_active and self.p.grid_update_interval > 0 and \
           (len(self) - self.last_bounds_update_bar) >= self.p.grid_update_interval:

            self.log.info(f"{log_prefix} Re-evaluando límites del grid (Intervalo: {self.p.grid_update_interval} barras)")
            # Calcular nuevos límites tentativos (sin activar nada aún)
            new_upper_tentative, new_lower_tentative = self._determine_bounds_internal()

            if new_upper_tentative is not None and new_lower_tentative is not None:
                # Calcular centro y ancho actuales y nuevos para comparar
                current_center = (self.active_upper_bound + self.active_lower_bound) / 2.0
                current_width = self.active_upper_bound - self.active_lower_bound
                new_center = (new_upper_tentative + new_lower_tentative) / 2.0
                new_width = new_upper_tentative - new_lower_tentative

                # Calcular cambios porcentuales (manejar división por cero)
                center_change_pct = abs(new_center - current_center) / current_center * 100.0 if abs(current_center) > 1e-9 else float('inf')
                width_change_pct = abs(new_width - current_width) / current_width * 100.0 if abs(current_width) > 1e-9 else float('inf')

                self.log.debug(f"{log_prefix} Re-eval: Centro Act={current_center:.{self.p.price_precision}f}, Nuevo={new_center:.{self.p.price_precision}f} (%Cambio={center_change_pct:.2f}%)")
                self.log.debug(f"{log_prefix} Re-eval: Ancho Act={current_width:.{self.p.price_precision}f}, Nuevo={new_width:.{self.p.price_precision}f} (%Cambio={width_change_pct:.2f}%)")

                # Comprobar si el cambio supera el umbral
                threshold = self.p.bound_change_threshold_percent
                if center_change_pct > threshold or width_change_pct > threshold:
                    self.log.warning(f"{log_prefix} Cambio límites ({center_change_pct:.1f}% Centro o {width_change_pct:.1f}% Ancho) > {threshold}%. DESACTIVANDO GRID para reajuste.")
                    # Desactivar grid actual para permitir reactivación con nuevos límites/niveles
                    self._desactivate_grid_safely(reason=f"Re-eval Interval (Cambio > {threshold}%)")
                    # Nota: El grid intentará reactivarse en la SIGUIENTE barra si las condiciones se cumplen
                    # Es importante retornar aquí para no intentar la lógica de mantenimiento en esta misma barra
                    return
                else:
                    # Si el cambio no supera el umbral, mantener el grid actual
                    self.log.debug(f"{log_prefix} Cambio límites dentro del umbral ({threshold}%). Grid se mantiene.")
                    # Actualizar contador para la próxima re-evaluación incluso si no hubo cambio
                    self.last_bounds_update_bar = len(self)
            else:
                # Si no se pudieron calcular nuevos límites, no hacer nada y loguear
                self.log.warning(f"{log_prefix} No se pudieron determinar nuevos límites tentativos durante re-evaluación. Grid se mantiene.")
                # Actualizar contador para evitar reintentos constantes si hay problema persistente
                self.last_bounds_update_bar = len(self)

        # --- 4. Lógica de Activación (Si Grid Inactivo) ---
        self.log.critical(f"!!! NEXT BAR {len(self)} - ANTES Lógica Activación (Activo: {self.grid_active}) !!!")
        if not self.grid_active:
            self.log.critical(f"!!! NEXT BAR {len(self)} - ENTRANDO Bloque Activación !!!") # Log de entrada al bloque

            # a. Determinar qué filtros se usarán en esta activación
            self._determine_active_filters()
            self.log.debug(f"{log_prefix} Intentando activar grid...")

            # b. Chequear Rango Base (ADX y BBW%)
            range_conditions_met = self.check_range_conditions()

            if range_conditions_met:
                # c. Chequear Filtros Adicionales (Tendencia, Volumen - si están activos)
                additional_filters_met = self._check_additional_filters_internal()

                if additional_filters_met:
                    # d. Chequear Filtro de Predicción (si está activo)
                    prediction_ok = True # Asumir OK si el filtro no está activo
                    if self._active_use_prediction_filter:
                        self.log.debug(f"{log_prefix} Filtro Predicción ACTIVO. Obteniendo datos...")
                        hist_df = self._get_historical_data_for_prediction()
                        if hist_df is not None and not hist_df.empty:
                            self.log.debug(f"{log_prefix} Llamando tu_funcion_de_prediccion con DF de {len(hist_df)} filas...")
                            try:
                                pred_result = self.tu_funcion_de_prediccion(hist_df)
                                # Interpretar resultado: True permite, False/None bloquean
                                prediction_ok = pred_result is True
                                self.log.info(f"{log_prefix} Resultado predicción: {pred_result} -> Permitido: {prediction_ok}")
                            except Exception as e_pred:
                                self.log.error(f"{log_prefix} EXCEPCION durante tu_funcion_de_prediccion: {e_pred}", exc_info=True)
                                prediction_ok = False # Bloquear en caso de error
                        else:
                            self.log.warning(f"{log_prefix} No se obtuvieron datos históricos válidos para predicción. BLOQUEANDO.")
                            prediction_ok = False # Bloquear si no hay datos

                    if prediction_ok:
                        # e. Determinar Límites U/L (solo si todas las condiciones anteriores OK)
                        self.log.debug(f"{log_prefix} Todas condiciones previas OK. Determinando límites...")
                        new_upper, new_lower = self._determine_bounds_internal()

                        if new_upper is not None and new_lower is not None:
                            # f. Calcular Niveles del Grid
                            self.log.debug(f"{log_prefix} Límites OK (L={new_lower}, U={new_upper}). Calculando niveles...")
                            levels = calculate_grid_levels(upper_bound=new_upper, lower_bound=new_lower,
                                                           num_grids=self.p.num_grids, mode=self.p.grid_mode,
                                                           price_precision=self.p.price_precision)

                            if levels and len(levels) >= 2:
                                # g. ¡Activar Grid! (Establecer estado)
                                self.log.warning(f"{log_prefix} === ACTIVANDO GRID ===")
                                print(f"INFO PRINT: {log_prefix} === ACTIVANDO GRID === L={new_lower:.{self.p.price_precision}f} U={new_upper:.{self.p.price_precision}f}")
                                self.grid_active = True
                                self.active_upper_bound = new_upper
                                self.active_lower_bound = new_lower
                                self.grid_levels = levels
                                # Guardar capital inicial justo ANTES de colocar órdenes
                                self.grid_initial_capital_usd = self.broker.getvalue()
                                self.last_bounds_update_bar = len(self) # Marcar barra de activación/última actualización
                                self.log.info(f"{log_prefix} Capital inicial registrado: ${self.grid_initial_capital_usd:,.2f}")

                                # h. Colocar Órdenes Iniciales
                                current_price = self.data_close[0]
                                num_placed = self._place_initial_grid_orders(current_price)
                                self.log.warning(f"{log_prefix} === GRID ACTIVADO ({len(levels)} niveles, {num_placed} órdenes colocadas) ===")
                                # Salir de next después de activar para evitar lógica de mantenimiento en la misma barra
                                return

                            else: # Falló cálculo de niveles
                               self.log.error(f"{log_prefix} Falló cálculo de niveles (Resultado: {levels}). No se activa grid.")
                        else: # Falló determinación de límites
                           self.log.debug(f"{log_prefix} Falló determinación de límites. No se activa grid.")
                    else: # Falló filtro de predicción
                       self.log.debug(f"{log_prefix} Filtro Predicción falló o bloqueó. No se activa grid.")
                else: # Fallaron filtros adicionales
                   self.log.debug(f"{log_prefix} Filtros adicionales fallaron. No se activa grid.")
            else: # Fallaron condiciones de rango base
               self.log.debug(f"{log_prefix} Condiciones de rango base no cumplidas. No se activa grid.")

        # --- 5. Mantenimiento (Si Grid Activo) ---
        # Esta lógica solo se ejecuta si el grid está activo y NO se acaba de activar en esta barra
        self.log.critical(f"!!! NEXT BAR {len(self)} - ANTES Mantenimiento (Activo: {self.grid_active}) !!!")
        if self.grid_active:
            current_price = self.data_close[0]
            if not np.isfinite(current_price):
                 self.log.warning(f"{log_prefix} Precio actual inválido ({current_price}). Omitiendo mantenimiento.")
                 return # Salir si el precio no es válido

            # a. Check Stop Loss Global (respecto a límites U/L)
            sl_margin_abs = (self.active_upper_bound - self.active_lower_bound) * (self.p.stop_loss_percent / 100.0)
            sl_threshold_price_long = self.active_lower_bound - sl_margin_abs
            sl_threshold_price_short = self.active_upper_bound + sl_margin_abs
            sl_triggered = False

            # Solo chequear SL si hay posición abierta
            if self.position:
                if self.position.size > 0 and current_price <= sl_threshold_price_long:
                     self.log.warning(f"{log_prefix} SL GLOBAL (LONG) ACTIVADO: Precio ({current_price:.{self.p.price_precision}f}) <= Límite Inf ({self.active_lower_bound:.{self.p.price_precision}f}) - Margen ({sl_margin_abs:.{self.p.price_precision}f}) = SL ({sl_threshold_price_long:.{self.p.price_precision}f})")
                     sl_triggered = True
                elif self.position.size < 0 and current_price >= sl_threshold_price_short:
                     self.log.warning(f"{log_prefix} SL GLOBAL (SHORT) ACTIVADO: Precio ({current_price:.{self.p.price_precision}f}) >= Límite Sup ({self.active_upper_bound:.{self.p.price_precision}f}) + Margen ({sl_margin_abs:.{self.p.price_precision}f}) = SL ({sl_threshold_price_short:.{self.p.price_precision}f})")
                     sl_triggered = True

                if sl_triggered:
                     self._desactivate_grid_safely(reason="Stop Loss Global (Límites)")
                     return # Salir después de desactivar

            # b. Check Take Profit Global (respecto a capital inicial)
            if self.grid_initial_capital_usd > 1e-9: # Asegurar que el capital inicial es válido
                 current_value = self.broker.getvalue()
                 profit = current_value - self.grid_initial_capital_usd
                 profit_percent = (profit / self.grid_initial_capital_usd) * 100.0
                 tp_threshold_percent = self.p.take_profit_percent
                 tp_triggered = False

                 if profit_percent >= tp_threshold_percent:
                      self.log.warning(f"{log_prefix} TP GLOBAL ACTIVADO: Ganancia ({profit_percent:.2f}%) >= TP ({tp_threshold_percent:.2f}%)")
                      tp_triggered = True

                 if tp_triggered:
                      self._desactivate_grid_safely(reason="Take Profit Global (Capital)")
                      return # Salir después de desactivar
            else:
                 # No se registró capital inicial válido, no se puede chequear TP
                 if len(self) == self.last_bounds_update_bar + 1 : # Loguear solo una vez post-activación
                      self.log.warning(f"{log_prefix} Capital inicial del grid ({self.grid_initial_capital_usd}) inválido. No se puede chequear TP global.")

            # (Podrías añadir más lógica de mantenimiento aquí si es necesario,
            #  como chequeos de órdenes individuales que no se llenan, etc.)

        # Log Crítico de fin normal de la ejecución de next
        self.log.critical(f"!!! NEXT BAR {len(self)} - FIN NORMAL DE NEXT !!!")

    # --- Fin del método next ---

# --- Fin Parte 10/13 ---
# -*- coding: utf-8 -*-
# ============================================================================================
# PARTE 11/13: Helpers: Historial Predicción, Eliminar Pendientes, Desactivar Grid (vD+DX v5)
# ============================================================================================
# (Continuacion de la clase GridBacktraderStrategyV5 desde Parte 10)

    # --- Helper: Obtiene datos históricos como DataFrame para la predicción ---
    def _get_historical_data_for_prediction(self) -> pd.DataFrame | None:
        """
        Extrae las últimas N barras (definidas por `prediction_lookback`)
        del datafeed principal y las devuelve como un DataFrame de Pandas.
        Necesario para pasar datos a funciones de predicción externas (como TF).

        Returns:
            pd.DataFrame or None: DataFrame con columnas Open, High, Low, Close,
                                 Volume y DatetimeIndex, o None si no hay
                                 suficientes datos o error.
        """
        log_prefix = "[_get_hist_pred]"
        try:
            # Obtener el número de barras a mirar hacia atrás desde los parámetros
            lookback = int(self.p.prediction_lookback)

            # Validar que el lookback sea positivo
            if lookback <= 0:
                self.log.warning(f"{log_prefix} prediction_lookback ({lookback}) debe ser > 0.")
                return None

            # Obtener la longitud actual de los datos disponibles
            current_len = len(self.data_main)

            # Verificar si ya tenemos suficientes barras (strict inequality needed for lookback)
            if current_len < lookback:
                # Loguear solo si es necesario (puede ser muy verboso)
                # self.log.debug(f"{log_prefix} Datos insuficientes ({current_len} < {lookback})")
                return None # No hay suficientes datos

            # Usar .get() para extraer los últimos 'lookback' valores como arrays numpy
            # ago=-1 incluye la barra actual que se está procesando en next()
            # size=lookback pide exactamente esa cantidad
            dates_num = self.data_datetime.get(ago=-1, size=lookback)
            opens = self.data_open.get(ago=-1, size=lookback)
            highs = self.data_high.get(ago=-1, size=lookback)
            lows = self.data_low.get(ago=-1, size=lookback)
            closes = self.data_close.get(ago=-1, size=lookback)
            volumes = self.data_volume.get(ago=-1, size=lookback)

            # Doble chequeo de longitud (puede ser menor al inicio del backtest si .get no devuelve todo)
            if len(dates_num) < lookback:
                self.log.warning(f"{log_prefix} .get() devolvió menos datos ({len(dates_num)}) que los solicitados ({lookback}).")
                return None

            # Convertir fechas numéricas de backtrader a objetos datetime
            # Esto puede ser un cuello de botella si se hace muy frecuentemente
            # Usar list comprehension para eficiencia
            datetime_index = [bt.num2date(d) for d in dates_num]

            # Crear el DataFrame de Pandas
            df_hist = pd.DataFrame({
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': closes,
                'Volume': volumes
            }, index=pd.DatetimeIndex(datetime_index)) # Usar fechas como índice

            # Verificar consistencia final del DataFrame (aunque el chequeo anterior debería bastar)
            if len(df_hist) != lookback:
                self.log.warning(f"{log_prefix} Longitud DF histórico final ({len(df_hist)}) != lookback ({lookback}).")
                # Decidir si devolver None o el DF parcial. Devolver None es más seguro.
                return None

            # Loguear éxito (opcional, puede ser verboso)
            # self.log.debug(f"{log_prefix} DataFrame histórico ({len(df_hist)}x{len(df_hist.columns)}) creado OK.")
            return df_hist

        except OverflowError as e_over:
            # Error común con fechas inválidas en num2date
            self.log.error(f"{log_prefix} OverflowError convirtiendo fechas num2date: {e_over}. ¿Datos corruptos?", exc_info=False)
            return None
        except Exception as e:
            # Capturar cualquier otro error inesperado
            self.log.error(f"{log_prefix} Error inesperado obteniendo datos históricos: {e}", exc_info=True)
            return None
    # --- Fin _get_historical_data_for_prediction ---


    # --- Helper para quitar orden de pendientes ---
    def _remove_order_from_pending(self, order_ref, log_prefix_caller, reason):
        """
        Elimina de forma segura una orden del diccionario `self.pending_orders`.
        Loguea la acción y maneja posibles errores.
        """
        # Verificar si la referencia de la orden existe en el diccionario
        if order_ref in self.pending_orders:
            log_prefix=f"[_remove_pending|{order_ref}]" # Prefijo específico para este log
            self.log.debug(f"{log_prefix} Eliminando Ref {order_ref} de pendientes (Razón: {reason}) (Llamado desde: {log_prefix_caller})...")
            try:
                # Intentar eliminar la entrada del diccionario
                del self.pending_orders[order_ref]
                self.log.debug(f"{log_prefix} Eliminado OK.")
            except KeyError:
                # Aunque se verificó antes, podría haber una condición de carrera (raro sin hilos)
                self.log.warning(f"{log_prefix} Intento de eliminar Ref {order_ref}, pero ya no existía (KeyError).")
            except Exception as e:
                # Capturar cualquier otro error inesperado
                self.log.error(f"{log_prefix} EXCEPCION al eliminar de pending_orders: {e}", exc_info=True)
        # else:
            # Opcional: Loguear si se intenta quitar una orden que no estaba
            # self.log.debug(f"[_remove_pending|{order_ref}] Intento de eliminar Ref {order_ref} (Razón: {reason}), pero no estaba en pendientes.")
    # --- Fin _remove_order_from_pending ---


    # --- Helper para desactivar el grid ---
    def _desactivate_grid_safely(self, reason=""):
        """
        Procedimiento seguro para desactivar el grid:
        1. Cierra posición abierta (si existe).
        2. Cancela todas las órdenes pendientes.
        3. Resetea variables de estado del grid.
        """
        try:
            dt_obj=self.dt();
            dt_str=dt_obj.strftime('%Y-%m-%d %H:%M:%S') if isinstance(dt_obj, datetime.datetime) else str(dt_obj)
        except Exception:
            dt_str = f"Barra {len(self)}"
        log_prefix = f"[_desactivate_grid|{dt_str}]"

        # Si ya está inactivo, no hacer nada
        if not self.grid_active:
            self.log.debug(f"{log_prefix} Ignorando desactivación (grid ya inactivo).")
            return

        # Loguear inicio de desactivación
        self.log.warning(f"{log_prefix} === INICIANDO DESACTIVACIÓN GRID ===")
        self.log.warning(f"{log_prefix} Motivo: {reason}")
        print(f"INFO PRINT: {log_prefix} === INICIANDO DESACTIVACIÓN GRID (Motivo: {reason}) ===") # Mantener print si es útil

        # 1. Cerrar posición abierta (si existe y es significativa)
        min_pos_size = 10**-(self.p.amount_precision + 1) # Umbral pequeño para evitar cerrar polvo
        if self.position and abs(self.position.size) >= min_pos_size:
            pos_size_fmt = f"{self.position.size:.{self.p.amount_precision}f}"
            self.log.info(f"{log_prefix} Cerrando Posición: {pos_size_fmt}...")
            print(f"INFO PRINT: {log_prefix} Cerrando Posición: {pos_size_fmt}...")
            try:
                # self.close() crea una orden MARKET para cerrar la posición actual del datafeed principal
                close_order = self.close(data=self.data_main)
                self.log.debug(f"{log_prefix} Orden de cierre enviada (Ref: {close_order.ref if close_order else 'N/A'}). Esperando notificación...")
            except bt.OrderPending:
                # Si ya hay una orden de cierre pendiente (raro pero posible)
                self.log.warning(f"{log_prefix} Ya existe una orden pendiente al intentar cerrar.")
            except Exception as e:
                self.log.error(f"{log_prefix} EXCEPCIÓN al enviar orden de cierre: {e}", exc_info=True)
        else:
             self.log.debug(f"{log_prefix} Sin posición abierta significativa para cerrar.")

        # 2. Cancelar todas las órdenes pendientes del grid
        # Este helper se define más adelante
        cancelled_count = self._cancel_all_pending_orders()
        self.log.info(f"{log_prefix} {cancelled_count} solicitudes de cancelación enviadas para órdenes pendientes.")

        # 3. Resetear variables de estado del grid
        self.log.debug(f"{log_prefix} Reseteando estado interno del grid...")
        self.grid_active = False
        self.active_upper_bound = None
        self.active_lower_bound = None
        self.grid_levels = []
        self.grid_initial_capital_usd = 0.0 # Resetear capital inicial
        # Limpiar diccionarios de órdenes y reposiciones pendientes AHORA
        # (ya se enviaron cancelaciones, notify_order las limpiará eventualmente,
        # pero limpiamos aquí para asegurar estado consistente inmediato)
        self.pending_orders.clear()
        self.pending_repositions.clear()
        self.last_bounds_update_bar = -1 # Resetear contador para re-evaluación

        # Resetear flags de filtros activos
        self._active_use_trend_filter = False
        self._active_use_volume_filter = False
        self._active_use_prediction_filter = False
        self.log.debug(f"{log_prefix} Estado interno reseteado.")

        # Loguear fin de desactivación
        self.log.warning(f"{log_prefix} === GRID DESACTIVADO ===")
        print(f"INFO PRINT: {log_prefix} === GRID DESACTIVADO ===")
    # --- Fin _desactivate_grid_safely ---

# --- Fin Parte 11/13 ---
# -*- coding: utf-8 -*-
# ============================================================================================
# PARTE 12/13: Helpers Restantes (Checks, Bounds, Sizing, Placing, Canceling) (vD+DX v5)
# ============================================================================================
# (Continuacion de la clase GridBacktraderStrategyV5 desde Parte 11)

    # --- Helper: Verifica condiciones de rango base ---
    def check_range_conditions(self):
        """
        Verifica condiciones de rango base (ADX < thr, BBW% < thr).
        Es la primera condición para activar el grid.

        Returns:
            bool: True si se cumplen las condiciones, False si no o hay error.
        """
        try:
            # Obtener timestamp seguro
            dt_obj=self.dt();
            dt_str=dt_obj.strftime('%Y-%m-%d %H:%M:%S') if isinstance(dt_obj,datetime.datetime) else str(dt_obj)
        except Exception:
            dt_str = f"Barra {len(self)}"
        log_prefix=f"[_check_range|{dt_str}]" # Prefijo específico

        try:
            # Verificar que los indicadores tengan datos suficientes
            if len(self.adx.lines.adx) == 0 or len(self.bbw_percent) == 0:
                self.log.debug(f"{log_prefix} Indicadores ADX/BBW% no listos (warm-up?).")
                return False # No se cumplen si no hay datos

            # Obtener valores actuales de los indicadores
            adx_value = self.adx.lines.adx[0]
            bbw_percent_value = self.bbw_percent[0]

            # Verificar que los valores sean números válidos (no NaN o Infinito)
            if not (np.isfinite(adx_value) and np.isfinite(bbw_percent_value)):
                self.log.warning(f"{log_prefix} Valores indicadores inválidos (NaN/inf): ADX={adx_value}, BBW%={bbw_percent_value}")
                return False # No se cumplen si son inválidos

            # Comparar con los umbrales definidos en los parámetros
            adx_ok = adx_value < self.p.adx_threshold
            bbw_ok = bbw_percent_value < self.p.bbw_threshold_percent
            range_ok = adx_ok and bbw_ok # Ambas deben cumplirse

            # Loguear siempre el resultado detallado
            self.log.debug(f"{log_prefix} Check Rango: ADX={adx_value:.2f} (<{self.p.adx_threshold}? {adx_ok}), "
                           f"BBW%={bbw_percent_value:.2f}% (<{self.p.bbw_threshold_percent:.2f}%? {bbw_ok}) -> Rango OK: {range_ok}")
            return range_ok # Devuelve True si ambas OK, False si no

        except IndexError:
            # Capturar error si se accede a [0] antes de tiempo (warm-up)
            self.log.warning(f"{log_prefix} IndexError accediendo a ADX/BBW% (warm-up?)")
            return False
        except Exception as e:
            # Capturar cualquier otro error inesperado
            self.log.error(f"{log_prefix} EXCEPCION inesperada verificando condiciones de rango: {e}", exc_info=True)
            return False # Asumir que no se cumplen en caso de error
    # --- Fin check_range_conditions ---


    # --- Helper: Verifica filtros adicionales ---
    def _check_additional_filters_internal(self):
        """
        Verifica filtros opcionales (Tendencia, Volumen) SI están activos
        según los flags _active_use_* actualizados por _determine_active_filters.

        Returns:
            bool: True si todos los filtros ACTIVOS se cumplen, False si alguno falla.
        """
        try:
            dt_obj=self.dt();
            dt_str=dt_obj.strftime('%Y-%m-%d %H:%M:%S') if isinstance(dt_obj, datetime.datetime) else str(dt_obj)
        except Exception:
            dt_str = f"Barra {len(self)}"
        log_prefix=f"[_check_filters|{dt_str}]"

        # --- Filtro de Tendencia ---
        if self._active_use_trend_filter:
            self.log.debug(f"{log_prefix} Filtro Tendencia ACTIVO. Verificando...")
            trend_ok=False
            try: # Try específico para el filtro de tendencia
                if len(self.sma_trend.lines.sma) > 0 and len(self.data_close) > 0:
                    sma_value = self.sma_trend.lines.sma[0]
                    close_price = self.data_close[0]
                    if np.isfinite(sma_value) and np.isfinite(close_price) and sma_value > 1e-9: # Evitar división por cero
                        deviation_pct = abs(close_price - sma_value) / sma_value * 100.0
                        max_dev = self.p.trend_filter_max_deviation
                        trend_ok = deviation_pct <= max_dev
                        self.log.debug(f"{log_prefix} Trend Check: Dev={deviation_pct:.2f}% (<= {max_dev:.2f}%? {trend_ok})")
                    else:
                        self.log.warning(f"{log_prefix} Trend Check: SMA ({sma_value}) / Close ({close_price}) inválido o SMA cero.")
                        trend_ok = False # Fallar si los datos no son válidos
                else:
                    self.log.debug(f"{log_prefix} Trend Check: SMA/Close no listos (warm-up?).")
                    trend_ok = False
            except IndexError:
                self.log.warning(f"{log_prefix} Trend Check: IndexError (warm-up?)."); trend_ok=False
            except Exception as e:
                self.log.error(f"{log_prefix} EXCEPCION Filtro Tendencia: {e}", exc_info=True); trend_ok=False

            if not trend_ok:
                self.log.debug(f"{log_prefix} ---> Filtro Tendencia: FALLÓ."); return False # Si falla, salir temprano
            self.log.debug(f"{log_prefix} ---> Filtro Tendencia: OK.")


        # --- Filtro de Volumen ---
        if self._active_use_volume_filter:
            self.log.debug(f"{log_prefix} Filtro Volumen ACTIVO. Verificando...")
            volume_ok=False
            try: # Try específico para el filtro de volumen
                if len(self.avg_volume.lines.sma) > 0 and len(self.data_volume) > 0:
                    current_vol = self.data_volume[0]
                    avg_vol_value = self.avg_volume.lines.sma[0]
                    if np.isfinite(current_vol) and np.isfinite(avg_vol_value):
                        if avg_vol_value >= 0: # Permitir volumen promedio cero
                            min_mult = self.p.volume_filter_min_mult
                            required_volume = avg_vol_value * min_mult
                            volume_ok = current_vol >= required_volume
                            # Caso especial: si ambos volúmenes son efectivamente cero, permitirlo
                            if avg_vol_value < 1e-9 and current_vol < 1e-9:
                                volume_ok = True
                            self.log.debug(f"{log_prefix} Vol Check: Vol={current_vol:.2f}, AvgVol({self.p.volume_filter_lookback})={avg_vol_value:.2f}, Mult={min_mult:.2f} -> Req>={required_volume:.2f}? OK:{volume_ok}")
                        else:
                            self.log.warning(f"{log_prefix} Vol Check: AvgVol negativo? ({avg_vol_value}). Filtro falla.")
                            volume_ok = False
                    else:
                         self.log.warning(f"{log_prefix} Vol Check: Vol/AvgVol inválido (NaN/inf?). Filtro falla.")
                         volume_ok = False
                else:
                    self.log.debug(f"{log_prefix} Vol Check: AvgVol/Volume no listos (warm-up?).")
                    volume_ok = False
            except IndexError:
                self.log.warning(f"{log_prefix} Vol Check: IndexError (warm-up?)."); volume_ok=False
            except Exception as e:
                self.log.error(f"{log_prefix} EXCEPCION Filtro Volumen: {e}", exc_info=True); volume_ok=False

            if not volume_ok:
                self.log.debug(f"{log_prefix} ---> Filtro Volumen: FALLÓ."); return False # Si falla, salir temprano
            self.log.debug(f"{log_prefix} ---> Filtro Volumen: OK.")

        # Si llegamos aquí, todos los filtros activos pasaron (o no había filtros activos)
        self.log.debug(f"{log_prefix} Filtros Adicionales: OK (o inactivos).")
        return True
    # --- Fin _check_additional_filters_internal ---


    # --- Helper: Determina límites U/L ---
    def _determine_bounds_internal(self):
        """
        Determina límites Upper/Lower según el método elegido en params.
        Realiza validaciones y ajustes.

        Returns:
            tuple (float, float) | tuple (None, None): (Upper, Lower) o (None, None) si hay error.
        """
        try:
            dt_obj=self.dt();
            dt_str=dt_obj.strftime('%Y-%m-%d %H:%M:%S') if isinstance(dt_obj,datetime.datetime) else str(dt_obj)
        except Exception:
            dt_str = f"Barra {len(self)}"
        log_prefix=f"[_determine_bounds|{dt_str}]"
        method=self.p.bounds_method
        upper, lower = None, None
        min_pos_price = 10**(-self.p.price_precision) # Tick size
        center=np.nan
        calculated_width=np.nan

        try: # Try principal para determinar límites
            if len(self.data_close)==0:
                 self.log.warning(f"{log_prefix} Close no listo (warm-up?)."); return None, None
            current_price = self.data_close[0]
            if not np.isfinite(current_price):
                 self.log.warning(f"{log_prefix} Close actual inválido ({current_price})."); return None, None

            # Calcular U/L base según método
            if method == 'ATR':
                if len(self.atr) == 0: self.log.warning(f"{log_prefix} ATR no listo."); return None, None
                atr_v = self.atr[0]
                mult = self.p.atr_multiplier
                if not np.isfinite(atr_v) or atr_v <= 1e-12: # ATR debe ser positivo
                    self.log.error(f"{log_prefix} ATR inválido ({atr_v})."); print(f"ERROR PRINT: {log_prefix} ATR inválido ({atr_v})."); return None, None
                center = current_price
                calculated_width = atr_v * mult
                upper = center + calculated_width / 2.0
                lower = center - calculated_width / 2.0
                self.log.debug(f"{log_prefix} ATR: Ctr={center:.{self.p.price_precision}f}, ATR({self.p.atr_period})={atr_v:.{self.p.price_precision}f}, Mult={mult:.2f} -> W={calculated_width:.{self.p.price_precision}f}")

            elif method == 'BB':
                bb = self.bb_bounds.lines
                if len(bb.top) == 0: self.log.warning(f"{log_prefix} BB (Bounds) no listo."); return None, None
                bb_t = bb.top[0]; bb_b = bb.bot[0]; bb_m = bb.mid[0]
                if not(np.isfinite(bb_t) and np.isfinite(bb_b) and np.isfinite(bb_m)):
                    self.log.error(f"{log_prefix} BB inválidos (T={bb_t}, M={bb_m}, B={bb_b})."); print(f"ERROR PRINT: {log_prefix} BB inválidos."); return None, None
                bb_w = max(0, bb_t - bb_b) # Ancho no puede ser negativo
                mult = self.p.bb_multiplier_bounds
                calculated_width = bb_w * mult
                center = bb_m # Usar Mid como centro
                upper = center + calculated_width / 2.0
                lower = center - calculated_width / 2.0
                self.log.debug(f"{log_prefix} BB: Mid={center:.{self.p.price_precision}f}, WidthBB={bb_w:.{self.p.price_precision}f}, Mult={mult:.2f} -> W={calculated_width:.{self.p.price_precision}f}")

            else:
                self.log.error(f"{log_prefix} bounds_method '{method}' desconocido."); return None, None

            # --- Validaciones y Ajustes Finales ---
            req_sep = min_pos_price / 2.0

            # 1. Ajustar L si es <= 0 (o muy cercano)
            if lower < min_pos_price:
                self.log.warning(f"{log_prefix} L calc ({lower:.{self.p.price_precision}f}) < {min_pos_price}. Ajustando L a {min_pos_price} y recalculando U.")
                lower = min_pos_price
                # Recalcular U basado en el ancho original para mantenerlo, si es posible
                adj_w = max(calculated_width if np.isfinite(calculated_width) and calculated_width > 0 else 0,
                            min_pos_price * max(1, self.p.num_grids)) # Ancho mínimo basado en ticks*grids
                upper = lower + adj_w

            # 2. Asegurar U > L + sep (después del ajuste de L si hubo)
            if upper <= lower + req_sep:
                 self.log.warning(f"{log_prefix} U ({upper:.{self.p.price_precision}f}) <= L ({lower:.{self.p.price_precision}f}) + Sep. Ajustando U para separación mínima.")
                 print(f"ADVERTENCIA PRINT: {log_prefix} U ({upper:.{self.p.price_precision}f}) <= L ({lower:.{self.p.price_precision}f}) + Sep. Ajustando U.")
                 num_int = max(1, self.p.num_grids)
                 min_w = min_pos_price * num_int # Ancho mínimo requerido para N grids
                 final_w = max(calculated_width if np.isfinite(calculated_width) and calculated_width > 1e-9 else 0, min_w, min_pos_price)
                 upper = lower + final_w
                 if upper <= lower + req_sep: # Re-chequear después del ajuste
                     self.log.error(f"{log_prefix} Fallo ajuste U > L+Sep. U={upper:.{self.p.price_precision}f}, L={lower:.{self.p.price_precision}f}")
                     return None, None

            # 3. Redondear a la precisión final
            low_f = round(lower, self.p.price_precision)
            up_f = round(upper, self.p.price_precision)

            # 4. Última verificación post-redondeo
            if low_f >= up_f - req_sep:
                 self.log.warning(f"{log_prefix} L_f({low_f}) >= U_f({up_f})-Sep post-rnd. Ajustando U_f.")
                 print(f"ADVERTENCIA PRINT: {log_prefix} L_f({low_f})>=U_f({up_f})-Sep post-rnd. Ajustando U_f.")
                 up_f = round(low_f + min_pos_price, self.p.price_precision) # Asegurar Uf = Lf + 1 tick
                 if low_f >= up_f - req_sep: # Chequeo final
                      self.log.error(f"{log_prefix} Fallo SUPER-ajuste U>L post-rnd. U={up_f}, L={low_f}")
                      return None, None

            # Éxito
            center_str = f"{center:.{self.p.price_precision}f}" if np.isfinite(center) else 'N/A'
            calculated_width_str = f"{calculated_width:.{self.p.price_precision}f}" if np.isfinite(calculated_width) else 'N/A'
            self.log.debug(f"{log_prefix} DETALLES: Mtd={method}, Ctr={center_str}, CalcW={calculated_width_str} -> L_F={low_f:.{self.p.price_precision}f}, U_F={up_f:.{self.p.price_precision}f}")
            self.log.info(f"{log_prefix} Límites determinados OK: L=${low_f:.{self.p.price_precision}f}, U=${up_f:.{self.p.price_precision}f}")
            return up_f, low_f # Devuelve U, L (redondeados y validados)

        except IndexError:
            self.log.warning(f"{log_prefix} IndexError accediendo a indicadores para bounds (warm-up?).")
            return None, None
        except Exception as e:
            self.log.error(f"{log_prefix} EXCEPCION determinando bounds: {e}", exc_info=True)
            return None, None
    # --- Fin _determine_bounds_internal ---


    # --- Helper: Calcula Tamaño de Orden --- (IMPLEMENTADO DURANTE DEBUG)
    def _get_order_amount(self, price):
        """ Calcula el tamaño de la orden basado en los parámetros. """
        size = 0.0
        min_size = 10**(-self.p.amount_precision)
        log_prefix = "[_get_order_amount]" # Añadir prefijo para logs

        try:
            if self.p.dynamic_order_size_mode == 'PERCENT':
                equity = self.broker.getvalue() # Usar equity total
                size_usd = equity * (self.p.order_size_percent / 100.0)
                if price > 1e-12: # Evitar división por cero
                    size = size_usd / price
                else:
                    self.log.warning(f"{log_prefix} Precio ({price}) inválido para calcular tamaño PERCENT.")
                    size = 0.0
                self.log.debug(f"{log_prefix} Mode=PERCENT: Equity={equity:.2f}, %={self.p.order_size_percent:.2f}, SizeUSD={size_usd:.2f}, Price={price:.{self.p.price_precision}f} -> Size={size:.{self.p.amount_precision}f}")

            elif self.p.dynamic_order_size_mode == 'FIXED':
                size_usd = self.p.order_size_usd
                if price > 1e-12:
                    size = size_usd / price
                else:
                     self.log.warning(f"{log_prefix} Precio ({price}) inválido para calcular tamaño FIXED.")
                     size = 0.0
                self.log.debug(f"{log_prefix} Mode=FIXED: SizeUSD={size_usd:.2f}, Price={price:.{self.p.price_precision}f} -> Size={size:.{self.p.amount_precision}f}")

            else:
                 self.log.error(f"{log_prefix} dynamic_order_size_mode '{self.p.dynamic_order_size_mode}' desconocido.")
                 size = 0.0

            # Aplicar redondeo hacia abajo (truncar) a la precisión definida
            if size > 0:
                factor = 10**self.p.amount_precision
                size = math.floor(size * factor) / factor
                self.log.debug(f"{log_prefix} Size truncado a {self.p.amount_precision} dec: {size:.{self.p.amount_precision}f}")


            # Asegurar tamaño mínimo (si es mayor que 0 pero menor que el mínimo, hacerlo 0)
            if 0 < size < min_size:
                self.log.warning(f"{log_prefix} Tamaño calculado {size:.{self.p.amount_precision}f} < mínimo {min_size:.{self.p.amount_precision}f}. Ajustando a 0.")
                size = 0.0
            elif size < 0: # Seguridad extra
                 self.log.error(f"{log_prefix} Tamaño calculado negativo ({size}). Forzando a 0.")
                 size = 0.0

        except Exception as e:
            self.log.error(f"{log_prefix} EXCEPCIÓN calculando tamaño: {e}", exc_info=True)
            size = 0.0

        self.log.debug(f"{log_prefix} Tamaño final devuelto: {size:.{self.p.amount_precision}f}")
        return size
    # --- Fin _get_order_amount ---


    # --- Helper: Coloca órdenes iniciales del grid ---
    def _place_initial_grid_orders(self, current_price):
        """
        Coloca las órdenes Limit iniciales (BUY por debajo del precio, SELL por encima)
        en los niveles del grid calculados. Omite niveles muy cercanos al precio actual.

        Args:
            current_price (float): El precio actual para determinar qué niveles son BUY/SELL.

        Returns:
            int: Número de órdenes que se intentaron colocar.
        """
        try:
            dt_obj=self.dt();
            dt_str=dt_obj.strftime('%Y-%m-%d %H:%M:%S') if isinstance(dt_obj, datetime.datetime) else str(dt_obj)
        except Exception:
            dt_str = f"Barra {len(self)}"
        log_prefix=f"[_place_initial|{dt_str}]"

        # Verificar que tenemos datos necesarios para operar
        if not self.grid_levels or len(self.grid_levels)<2 or self.active_upper_bound is None or self.active_lower_bound is None:
            self.log.error(f"{log_prefix} No se pueden colocar órdenes iniciales: Faltan niveles (<2) o límites activos.")
            return 0 # No se colocaron órdenes

        self.log.info(f"{log_prefix} Colocando órdenes iniciales para {len(self.grid_levels)} niveles (Precio actual=${current_price:.{self.p.price_precision}f})...")
        placed_count=0 # Contador de órdenes colocadas
        min_size=10**(-self.p.amount_precision)
        min_tick=10**(-self.p.price_precision)

        # Calcular espaciado promedio y tolerancia de proximidad
        num_int = len(self.grid_levels) - 1
        avg_spacing = (self.active_upper_bound - self.active_lower_bound) / num_int if num_int > 0 else 0
        # Tolerancia: 10% del espacio promedio, o 2 ticks como mínimo
        prox_tol = max(min_tick * 2, avg_spacing * 0.10) if avg_spacing > 0 else min_tick * 2
        self.log.debug(f"{log_prefix} Tolerancia proximidad={prox_tol:.{self.p.price_precision}f}")

        # Iterar sobre cada nivel calculado
        for i, level_p in enumerate(self.grid_levels):
            price_diff = abs(level_p - current_price)
            self.log.debug(f"{log_prefix} Eval Lvl {i}: P={level_p:.{self.p.price_precision}f}, Diff={price_diff:.{self.p.price_precision}f}")

            # Omitir niveles demasiado cercanos al precio actual
            if price_diff < prox_tol:
                self.log.debug(f"{log_prefix} Lvl {i} OMITIDO (muy próximo al precio actual).")
                continue

            # Determinar lado (BUY si nivel < precio, SELL si nivel > precio)
            side = "BUY" if level_p < current_price else ("SELL" if level_p > current_price else None)

            if side: # Si no es igual al precio actual
                # Calcular tamaño de orden usando el helper
                amount = self._get_order_amount(level_p)
                self.log.debug(f"{log_prefix} Lvl {i}: Side={side}, P={level_p:.{self.p.price_precision}f}, SzPot={amount:.{self.p.amount_precision}f}, MinSz={min_size:.{self.p.amount_precision}f}")

                # Omitir si el tamaño es inválido (demasiado pequeño)
                if amount < min_size:
                    self.log.warning(f"{log_prefix} Lvl {i} OMITIDO (tamaño inválido/cero).")
                    continue

                # Colocar la orden límite
                try:
                    order_func = self.buy if side == "BUY" else self.sell
                    order = order_func(price=level_p, size=amount, exectype=bt.Order.Limit, data=self.data_main)

                    # Si la orden se creó correctamente (devuelve un objeto Order), registrarla en pendientes
                    if order and order.ref is not None: # Chequeo más robusto
                        self.pending_orders[order.ref] = {
                            'level_idx': i,
                            'side': side.lower(),
                            'price': level_p,
                            'size': amount,
                            'order': order, # Guardar referencia al objeto Order
                            'is_reposition_order': False # Marcar como orden original
                        }
                        placed_count+=1
                        self.log.info(f"{log_prefix} -> OK Ini {side} Lvl {i} @ ${level_p:.{self.p.price_precision}f} (Sz={amount:.{self.p.amount_precision}f}) Ref:{order.ref}.")
                    else:
                        # Error si la orden no se pudo crear/registrar
                        self.log.error(f"{log_prefix} Fallo crear/registrar orden {side} Lvl {i}. Orden devuelta: {order}")
                        print(f"ERROR PRINT: {log_prefix} Fallo crear/reg orden {side} Lvl {i}.")

                except Exception as e_place:
                    # Capturar error durante la colocación
                    self.log.error(f"{log_prefix} EXCEPCION orden {side} Lvl {i} @ ${level_p:.{self.p.price_precision}f}: {e_place}", exc_info=True)
                    print(f"ERROR CRITICO PRINT: {log_prefix} EXCEPCION orden {side} Lvl {i}: {e_place}")

        # Loguear resumen de colocación
        self.log.info(f"{log_prefix} Colocación inicial finalizada. Total colocadas: {placed_count}")
        print(f"INFO PRINT: {log_prefix} Órdenes iniciales: {placed_count}")
        return placed_count
    # --- Fin _place_initial_grid_orders ---


    # --- Helper: Cancela todas las órdenes pendientes ---
    def _cancel_all_pending_orders(self):
        """
        Intenta cancelar todas las órdenes activas registradas en `self.pending_orders`.
        Devuelve el número de solicitudes de cancelación enviadas.
        """
        try:
            dt_obj=self.dt();
            dt_str=dt_obj.strftime('%Y-%m-%d %H:%M:%S') if isinstance(dt_obj, datetime.datetime) else str(dt_obj)
        except Exception:
            dt_str = f"Barra {len(self)}"
        log_prefix = f"[_cancel_all|{dt_str}]" # Prefijo específico

        num_to_cancel = len(self.pending_orders)
        if num_to_cancel == 0:
            self.log.debug(f"{log_prefix} No hay órdenes pendientes para cancelar.")
            return 0 # No se enviaron cancelaciones

        self.log.info(f"{log_prefix} Solicitando cancelación de {num_to_cancel} órdenes pendientes...")
        print(f"INFO PRINT: {log_prefix} Cancelando {num_to_cancel} órdenes pendientes...")
        sent_reqs=0 # Contador de cancelaciones enviadas

        # Iterar sobre una copia de las claves para poder modificar el diccionario
        refs_to_proc = list(self.pending_orders.keys())
        for ref in refs_to_proc:
            order_data = self.pending_orders.get(ref)

            # Verificar que tenemos datos y un objeto Order válido
            if order_data and isinstance(order_data.get('order'), bt.Order):
                order = order_data['order']
                try:
                    # Solo cancelar si la orden está "viva" (no completada, cancelada, etc.)
                    if order.alive():
                        self.cancel(order) # Enviar solicitud de cancelación
                        sent_reqs+=1
                        self.log.debug(f"{log_prefix} Cancel Req enviada Ref {ref} (Status actual: {order.getstatusname()})")
                    else:
                        # Si no está viva, no se puede cancelar, solo limpiar el registro
                        self.log.debug(f"{log_prefix} Ref {ref} no 'viva' (St:{order.getstatusname()}). Limpiando registro de pendientes.")
                        # No es necesario llamar a remove aquí, se limpiará globalmente al final de _desactivate_grid_safely
                        # o individualmente en notify_order si llega una notificación final.
                        # _remove_order_from_pending(ref, log_prefix, f"No viva({order.getstatusname()})") # Opcional
                except Exception as e:
                    # Error al intentar cancelar
                    self.log.error(f"{log_prefix} EXCEPCION cancelando Ref {ref}: {e}", exc_info=True)
                    # Considerar limpiar registro igualmente si falla la cancelación
                    # _remove_order_from_pending(ref, log_prefix, f"Excep cancel: {e}") # Opcional
            else:
                # Si la entrada en pending_orders es inválida, limpiarla (aunque se hará clear() después)
                self.log.warning(f"{log_prefix} Ref {ref} sin datos/Order válidos en pendientes. Se limpiará.")
                # _remove_order_from_pending(ref, log_prefix, "Inválida/Sin Order") # Opcional

        # Loguear resumen de cancelaciones enviadas
        # Nota: pending_orders se limpia completamente en _desactivate_grid_safely después de llamar a esto
        self.log.info(f"{log_prefix} {sent_reqs} cancel reqs enviadas.")
        return sent_reqs
    # --- Fin _cancel_all_pending_orders ---

# --- Fin Parte 12/13 ---
# -*- coding: utf-8 -*-
# ============================================================================================
# PARTE 13/13: Funcion Predicción Placeholder, Fin Clase, Bloque Main (vD+DX v5 - runonce=False)
# ============================================================================================
# (Continuacion de la clase GridBacktraderStrategyV5 desde Parte 12)

    # --- Metodo de Prediccion (EJEMPLO TF - Placeholder) ---
    def tu_funcion_de_prediccion(self, dataframe_historico):
        """
        Lógica de predicción/filtrado. Devuelve True/False/None.
        EJEMPLO TF Simplificado. Necesita tu lógica real.
        Devuelve True si las condiciones (ROC, VolRatio, Z-Score) se cumplen,
        False si no se cumplen, None si hay error o datos insuficientes.
        """
        logger = self.log # Usar el logger de la instancia

        try:
            # Obtener timestamp seguro
            dt_obj=self.dt();
            dt_str=dt_obj.strftime('%Y-%m-%d %H:%M:%S') if isinstance(dt_obj, datetime.datetime) else str(dt_obj)
        except Exception:
            dt_str = f"Barra {len(self)}"
        log_prefix = f"[{self.__class__.__qualname__}|PredTF|{dt_str}]"

        # Verificar disponibilidad global de TF
        if not _tf_available:
            logger.warning(f"{log_prefix} TF no disponible. Filtro predicción BLOQUEADO (devuelve False).")
            return False # Bloquear si TF no está

        try: # Try principal de la función de predicción
            req_cols=['Close','Volume']; p = self.p # Columnas requeridas y alias a params

            # Validaciones de entrada del DataFrame
            if not isinstance(dataframe_historico, pd.DataFrame) or dataframe_historico.empty:
                logger.warning(f"{log_prefix} DataFrame histórico inválido o vacío. BLOQUEANDO (devuelve None).")
                return None
            if not all(c in dataframe_historico.columns for c in req_cols):
                logger.warning(f"{log_prefix} Faltan columnas requeridas {req_cols} en DF histórico. BLOQUEANDO (devuelve None).")
                return None

            # Obtener parámetros necesarios para los cálculos TF de ejemplo
            roc_p=getattr(p,'pred_roc_period',5)
            roc_thr=getattr(p,'pred_roc_threshold_pct',1.0)
            vol_s=getattr(p,'pred_vol_short_period',5)
            vol_l=getattr(p,'pred_vol_long_period',20)
            vol_max=getattr(p,'pred_vol_ratio_max',2.0)
            vol_min=getattr(p,'pred_vol_ratio_min',0.5)
            mrev_sma=getattr(p,'pred_mrev_sma_period',20)
            mrev_std=getattr(p,'pred_mrev_std_period',20)
            mrev_z=getattr(p,'pred_mrev_zscore_threshold',1.5)

            # Calcular lookback máximo requerido por los indicadores internos de esta función
            lookback_pred_internal = max(roc_p + 1, vol_l, mrev_sma, mrev_std) + 1 # Añadir margen
            if len(dataframe_historico) < lookback_pred_internal:
                logger.warning(f"{log_prefix} Datos insuficientes en DF ({len(dataframe_historico)} < {lookback_pred_internal} requeridos para cálculo TF). BLOQUEANDO (devuelve None).")
                return None

            # --- Función interna para cálculos TF ---
            # (Podría optimizarse con @tf.function si el rendimiento es crítico)
            def calc_tf(close_tensor, vol_tensor):
                eps = tf.constant(1e-12, dtype=tf.float32) # Epsilon para evitar división por cero
                close_tf = tf.cast(close_tensor, dtype=tf.float32)

                # ROC: Rate of Change
                p_now = close_tf[-1]
                p_then = close_tf[-1 - roc_p]
                roc = tf.where(tf.abs(p_then) < eps, 0.0, (p_now - p_then) / p_then * 100.0)

                # Volatility Ratio: stddev(short) / stddev(long)
                # Usar reduce_std que calcula stddev sobre toda la ventana (NO es rolling)
                std_s = tf.math.reduce_std(close_tf[-vol_s:])
                std_l = tf.math.reduce_std(close_tf[-vol_l:])
                vol_r = tf.where(std_l < eps, 1.0, std_s / std_l) # Evitar división por cero

                # Z-Score (Mean Reversion indicator): (price - sma) / stddev
                sma = tf.math.reduce_mean(close_tf[-mrev_sma:])
                std = tf.math.reduce_std(close_tf[-mrev_std:])
                z = tf.where(std < eps, 0.0, tf.abs(p_now - sma) / std) # Tomar valor absoluto

                return roc, vol_r, z
            # --- Fin función interna calc_tf ---

            # Preparar datos como tensores TF
            # Asegurar tomar suficientes filas para el lookback interno MÁXIMO de calc_tf
            n_rows_needed = lookback_pred_internal
            close_np = dataframe_historico['Close'].values[-n_rows_needed:].astype(np.float32)
            vol_np = dataframe_historico['Volume'].values[-n_rows_needed:].astype(np.float32) # No usado en este ejemplo calc_tf

            # Ejecutar cálculos TF
            roc_v, vol_r_v, z_v = calc_tf(tf.constant(close_np), tf.constant(vol_np))

            # Convertir resultados a float estándar de Python
            roc = float(roc_v.numpy())
            vol_r = float(vol_r_v.numpy())
            z = float(z_v.numpy())

            # Validar resultados numéricos (si falló el cálculo TF)
            if not(np.isfinite(roc) and np.isfinite(vol_r) and np.isfinite(z)):
                logger.warning(f"{log_prefix} Cálculos TF resultaron en NaN/Inf. ROC={roc}, VolR={vol_r}, Z={z}. BLOQUEANDO (devuelve False).")
                return False # Bloquear si los cálculos fallan

            logger.debug(f"{log_prefix} Vals TF: ROC={roc:.2f}%(Thr={roc_thr}%), VolR={vol_r:.2f}(Rng=[{vol_min},{vol_max}]), Z={z:.2f}(Thr={mrev_z})")

            # --- Aplicar condiciones del filtro de predicción ---
            # 1. ROC bajo (mercado no tendencial)
            if abs(roc) > roc_thr:
                logger.debug(f"{log_prefix} --- Pred FALLÓ (ROC |{roc:.2f}%| > {roc_thr:.2f}%) ---")
                return False # Bloquear si hay mucho movimiento reciente

            # 2. Ratio de Volatilidad en rango (ni muy explosivo ni muy muerto)
            if not(vol_min <= vol_r <= vol_max):
                logger.debug(f"{log_prefix} --- Pred FALLÓ (VolR {vol_r:.2f} no en [{vol_min:.2f},{vol_max:.2f}]) ---")
                return False # Bloquear si la volatilidad relativa está fuera de rango

            # 3. Z-Score bajo (precio no muy alejado de su media reciente -> potencial rango)
            if z > mrev_z:
                logger.debug(f"{log_prefix} --- Pred FALLÓ (Z {z:.2f} > {mrev_z:.2f}) ---")
                return False # Bloquear si el precio está muy extendido

            # Si pasa todos los filtros, PERMITIR activación del grid
            logger.info(f"{log_prefix} +++ Pred OK. PERMITIENDO +++")
            return True

        except Exception as e_pred_main:
            # Capturar error general de la predicción
            logger.error(f"{log_prefix} Error inesperado en lógica de predicción TF: {e_pred_main}", exc_info=True)
            print(f"ERROR PRINT: {log_prefix} Error predicción TF: {e_pred_main}")
            return None # Devolver None en caso de error para bloquear por seguridad

    # --- FIN DE LA CLASE STRATEGY ---
# --- Fin Clase GridBacktraderStrategyV5 ---

# ==========================================================================================================
# PARTE 13/13: Bloque Principal (`__main__`) Completo (vD+DX v5 - runonce=False)
# ==========================================================================================================

# ============================================================================================
# === BLOQUE PRINCIPAL DE EJECUCION (`if __name__ == '__main__':`) ===
# ============================================================================================
if __name__ == '__main__':

    # Añadir un print de verificación al inicio del main
    print("--- [Bloque Principal] Iniciando ejecución __main__ ---")
    logging.info("--- [Bloque Principal] Iniciando ejecución __main__ ---")

    # ==========================================
    # PARTE A: Config General y Carga Datos
    # ==========================================
    logging.info("\n"+"="*40+"\n--- BLOQUE PRINCIPAL: Configuración y Carga de Datos ---\n"+"="*40)
    print("\n--- [Bloque Principal] Configuración y Carga de Datos ---")

    start_cash = 10000.0
    commission = 0.001 # 0.1%
    slippage_percentage = 0.0002 # 0.02%

    # --- Configuración de Datos ---
    # <<< ¡¡¡ VERIFICAR NOMBRE Y RUTA DEL ARCHIVO CSV !!! >>>
    csv_filename=f'BTCUSDT-5m-2024-11-01_to_2025-04-14.csv' # Asegúrate que este archivo existe
    csv_filepath = csv_filename # Usar el nombre directamente si está en la misma carpeta
    data_timeframe = bt.TimeFrame.Minutes
    data_compression = 5
    split_ratio = 0.70 # 70% IS / 30% OOS

    logging.info(f"Config BT: Cash=${start_cash:,.2f}, Comm={commission*100:.3f}%, Slip={slippage_percentage*100:.3f}%")
    logging.info(f"Config Datos: Arch='{csv_filepath}', TF={bt.TimeFrame.getname(data_timeframe, data_compression)}/{data_compression}, Split={split_ratio*100:.0f}%")

    dataframe_is=None; dataframe_oos=None; all_data_df=None
    current_run_timestamp = datetime.datetime.now(lima_tz if _timezone_info_available else None)

    try:
        # --- Carga y Preprocesamiento ---
        print(f"TS inicio: {current_run_timestamp.isoformat()}")
        logging.info(f"TS inicio: {current_run_timestamp.isoformat()}")
        logging.info(f"Cargando: '{csv_filepath}'...")
        print(f"Cargando: '{csv_filepath}'...")
        if not os.path.exists(csv_filepath):
            logging.error(f"¡FATAL! Archivo no encontrado: '{csv_filepath}'")
            raise FileNotFoundError(f"Archivo no encontrado: {csv_filepath}")

        all_data_df = pd.read_csv(csv_filepath, index_col=0, parse_dates=True)
        logging.info(f"Cargado: {len(all_data_df)} filas.")

        # Renombrar columnas a formato Backtrader (Open, High, Low, Close, Volume) - case-insensitive
        col_map={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
        rename_map={c:col_map[c.lower()] for c in all_data_df.columns if c.lower() in col_map}
        all_data_df.rename(columns=rename_map, inplace=True)
        req_cols=['Open','High','Low','Close','Volume']
        missing=[c for c in req_cols if c not in all_data_df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas después de renombrar: {missing}")

        # Asegurar índice Datetime y orden
        if not isinstance(all_data_df.index, pd.DatetimeIndex):
            logging.warning("Índice no es DatetimeIndex. Convirtiendo...")
            all_data_df.index = pd.to_datetime(all_data_df.index, errors='coerce')
            all_data_df.dropna(subset=[all_data_df.index.name], inplace=True) # Eliminar filas donde la conversión de fecha falló
        if not all_data_df.index.is_monotonic_increasing:
            logging.warning("Índice no ordenado ascendentemente. Ordenando...")
            all_data_df.sort_index(inplace=True)

        # Limpieza de NaNs y conversión numérica
        rows_b = len(all_data_df)
        logging.debug("Convirtiendo columnas OHLCV a numérico y eliminando NaNs...")
        for col in req_cols:
             # errors='coerce' convertirá lo no numérico a NaN
             all_data_df[col] = pd.to_numeric(all_data_df[col], errors='coerce')
        all_data_df.dropna(subset=req_cols, inplace=True) # Eliminar filas con NaN en OHLCV
        rows_a = len(all_data_df)
        if rows_a < rows_b:
            logging.warning(f"Eliminadas {rows_b - rows_a} filas con NaNs/Inválidos en OHLCV.")

        if all_data_df.empty:
            raise ValueError("DataFrame vacío después del preprocesamiento.")

        logging.info(f"Preprocesamiento OK: {len(all_data_df)} filas válidas.")
        logging.info(f"Rango de datos: [{all_data_df.index.min()}] -> [{all_data_df.index.max()}]")
        print(f"Datos OK: {len(all_data_df)} filas. [{all_data_df.index.min()}] -> [{all_data_df.index.max()}]")

        # Dividir en In-Sample (IS) y Out-of-Sample (OOS)
        split_idx = int(len(all_data_df) * split_ratio)
        if not (0 < split_idx < len(all_data_df)): # Asegurar que el índice de split es válido
            raise ValueError(f"Índice de split inválido ({split_idx}) para longitud de datos {len(all_data_df)}")

        dataframe_is = all_data_df.iloc[:split_idx].copy()
        dataframe_oos = all_data_df.iloc[split_idx:].copy()
        # ---> INICIO DEBUG DATAFRAME_IS (objective_function) <---
        #print(f"\n{log_prefix} --- DEBUG DATAFRAME_IS ---")
        #print(f"{log_prefix} Tipo de dataframe_is: {type(dataframe_is)}")
        #if isinstance(dataframe_is, pd.DataFrame) and not dataframe_is.empty:
        #    print(f"{log_prefix} Shape de dataframe_is: {dataframe_is.shape}")
        #    # Usar try-except por si el índice no es datetime
        #    try:
        #        print(f"{log_prefix} Primeras 3 filas índice: \n{dataframe_is.head(3).index}")
        #        print(f"{log_prefix} Últimas 3 filas índice: \n{dataframe_is.tail(3).index}")
        #        print(f"{log_prefix} Rango Fechas IS Esperado: {dataframe_is.index.min()} -> {dataframe_is.index.max()}")
        #    except Exception as e_df_print:
        #         print(f"{log_prefix} Error al imprimir detalles del índice: {e_df_print}")
        #else:
         #   print(f"{log_prefix} dataframe_is no es un DataFrame válido o está vacío.")
        #print(f"{log_prefix} --- FIN DEBUG DATAFRAME_IS ---\n")
        # ---> FIN DEBUG DATAFRAME_IS <---

        if dataframe_is.empty or dataframe_oos.empty:
            raise ValueError("División IS/OOS resultó en DataFrames vacíos.")

        logging.info(f"Split OK: IS={len(dataframe_is)} ({dataframe_is.index.min()}->{dataframe_is.index.max()}), OOS={len(dataframe_oos)} ({dataframe_oos.index.min()}->{dataframe_oos.index.max()})")
        print(f"  > IS : {len(dataframe_is)} filas ({dataframe_is.index.min()}->{dataframe_is.index.max()})")
        print(f"  > OOS: {len(dataframe_oos)} filas ({dataframe_oos.index.min()}->{dataframe_oos.index.max()})")

    except FileNotFoundError as e_fnf:
        logging.error(f"¡ERROR FATAL! {e_fnf}", exc_info=False)
        print(f"¡ERROR FATAL! {e_fnf}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e_val:
         logging.error(f"¡ERROR FATAL en validación de datos/split!: {e_val}", exc_info=True)
         print(f"¡ERROR FATAL! {e_val}", file=sys.stderr)
         sys.exit(1)
    except Exception as e_load:
        logging.error(f"¡ERROR FATAL CARGA/PROCESADO DATOS!: {e_load}", exc_info=True)
        print(f"¡ERROR FATAL! {e_load}", file=sys.stderr)
        sys.exit(1)


    # ==========================================
    # PARTE B: Config Optimización Optuna
    # ==========================================
    logging.info("\n"+"="*40+"\n--- BLOQUE PRINCIPAL: Configuración Optimización Optuna ---\n"+"="*40)
    print("\n--- [Bloque Principal] Config Optuna ---")

    # Parámetros fijos que no se optimizarán
    fixed_params_for_objective = dict(
        grid_mode='LINEAR',                  # Modo de grid fijo
        dynamic_order_size_mode='PERCENT',   # Usar % del equity
        order_size_usd=10.0,                 # (Ignorado si mode=PERCENT)
        # Fijar params de BB para filtro de rango (si bounds_method=ATR)
        bb_period_range=20,                  # (Usado por BBW%)
        bb_stddev_range=2.0,                 # (Usado por BBW%)
        # Fijar params de BB para cálculo de límites (si bounds_method=BB)
        bb_period_bounds=20,
        bb_stddev_bounds=2.0,
        # Lookback fijo para la función de predicción
        prediction_lookback=100,
        # Precisiones fijas
        price_precision=8,
        amount_precision=8,
    )
    logging.info(f"Params FIJOS estrategia:\n{json.dumps(fixed_params_for_objective, indent=2)}")

    # Config Ejecución Optuna
    N_TRIALS = 1    # <<< MANTENER N_TRIALS=1 HASTA QUE FUNCIONE BIEN >>>
    N_JOBS = 1      # <<< MANTENER N_JOBS=1 HASTA QUE FUNCIONE BIEN >>> (Multiproceso puede complicar debug de DB/logs)
    R_SEED = 42     # Para reproducibilidad del sampler de Optuna
    
    logging.info(f"\nConfig Optuna: n_trials={N_TRIALS}, n_jobs={N_JOBS}, seed={R_SEED}")
    print(f"  > Config Optuna: {N_TRIALS} trials, {N_JOBS} jobs, Seed={R_SEED}")


    # =======================================================================
    # PARTE C: Función Objetivo Optuna
    # =======================================================================
    logging.info("\n" + "="*40 + "\n--- BLOQUE PRINCIPAL: Función Objetivo y Ejecución Optuna IS ---\n" + "="*40)
    print("\n--- [Bloque Principal] Optuna IS ---")

    obj_counter=0 # Contador simple para logs

    def objective_function(trial):
        """
        Función objetivo para Optuna.
        1. Sugiere hiperparámetros.
        2. Configura y ejecuta backtest In-Sample con esos parámetros.
           - La estrategia (notify_trade) escribe PnL y Barlen en una DB SQLite temporal.
        3. Lee los resultados de la DB SQLite.
        4. Calcula métricas (SQN principalmente).
        5. Devuelve un score (a minimizar) basado en SQN con penalizaciones.
        """
        global obj_counter
        obj_counter+=1
        # Identificador único para logs y DB de este trial
        run_id=f"Trial-{trial.number}_PID-{os.getpid()}"
        log_prefix=f"[ObjectiveOptuna-{run_id}]"
        logging.info(f"\n{log_prefix} {'='*10} INICIO Eval Trial #{trial.number} {'='*10}")

        # Inicializar variables
        opt_params={}; params={}
        score=1e12 # Score muy alto (malo) por defecto
        metric_src="Inicio/Fallo General" # Razón del score
        trades=0; pnl=np.nan; pnl_mean=np.nan; pnl_std=np.nan; sqn=np.nan
        err_info=None # Para guardar info de errores
        db_filename=f"temp_trial_{trial.number}_pid_{os.getpid()}_trades.db" # DB única por trial/proceso

        try: # TRY PRINCIPAL DEL OBJETIVO - Captura errores catastróficos
            if dataframe_is is None or dataframe_is.empty:
                raise ValueError("DataFrame In-Sample (dataframe_is) no disponible o vacío.")

            # --- 1. Setup DB Temporal SQLite ---
            logging.debug(f"{log_prefix} Preparando DB SQLite temporal: {db_filename}")
            # Borrar DB si existe de una ejecución anterior interrumpida
            if os.path.exists(db_filename):
                try:
                    os.remove(db_filename)
                    logging.debug(f"{log_prefix} DB temp anterior '{db_filename}' borrada.")
                except Exception as e_del:
                    # No crítico, pero puede indicar problemas de permisos o locking
                    logging.warning(f"{log_prefix} No se pudo borrar DB temp anterior '{db_filename}': {e_del}")

            # Crear la nueva DB y tabla (usar 'with' para manejo seguro)
            try:
                with sqlite3.connect(db_filename) as conn_setup:
                     # Usar TEXT para pnlcomm por si acaso, INTEGER para barlen
                    conn_setup.execute('CREATE TABLE trades (pnlcomm REAL NOT NULL, barlen INTEGER NOT NULL)')
                    conn_setup.commit()
                logging.debug(f"{log_prefix} Tabla 'trades' creada en DB {db_filename}")
            except sqlite3.Error as e_db_create:
                logging.error(f"{log_prefix} FATAL creando tabla SQLite: {e_db_create}",exc_info=True)
                # Si no se puede crear la DB, no se puede continuar el trial
                raise RuntimeError(f"Fallo creación DB ({e_db_create})") from e_db_create


            # --- 2. Sugerir Parámetros Optuna ---
            logging.debug(f"{log_prefix} Sugiriendo parámetros Optuna...")
            opt_params={
                # Grid
                'num_grids': trial.suggest_int('num_grids', 3, 9, step=2),
                'grid_update_interval': trial.suggest_int('grid_update_interval', 24, 240, step=24), # ~2h a 20h en 5min
                'bound_change_threshold_percent': trial.suggest_float('bound_change_threshold_percent', 30.0, 90.0),
                # Sizing
                'order_size_percent': trial.suggest_float('order_size_percent', 0.5, 5.0, log=True),
                # Risk
                'stop_loss_percent': trial.suggest_float('stop_loss_percent', 1, 5),
                'take_profit_percent': trial.suggest_float('take_profit_percent', 2.0, 10.0),
                # Range Filter
                'adx_period': trial.suggest_int('adx_period', 7, 30),
                'adx_threshold': trial.suggest_int('adx_threshold', 20, 40),
                'bbw_threshold_percent': trial.suggest_float('bbw_threshold_percent', 1, 10),
                # Bounds Method & Params
                'bounds_method': trial.suggest_categorical('bounds_method', ['ATR', 'BB']),
                'atr_period': trial.suggest_int('atr_period', 7, 28),
                'atr_multiplier': trial.suggest_float('atr_multiplier', 1.0, 3.5),
                'bb_multiplier_bounds': trial.suggest_float('bb_multiplier_bounds', 0.8, 3.0), # Multiplicador sobre ancho de BB
                # Repositioning
                'reposition_dist_ticks': trial.suggest_int('reposition_dist_ticks', 1, 10),
                'reposition_timeout_bars': trial.suggest_int('reposition_timeout_bars', 3, 30),
                # Filter Activation
                'use_trend_filter': trial.suggest_categorical('use_trend_filter', [True, False]),
                'use_volume_filter': trial.suggest_categorical('use_volume_filter', [True, False]),
                'use_prediction_filter': trial.suggest_categorical('use_prediction_filter', [True, False]),
                # Trend Filter Params
                'trend_filter_ma_period': trial.suggest_int('trend_filter_ma_period', 50, 200, step=10),
                'trend_filter_max_deviation': trial.suggest_float('trend_filter_max_deviation', 0.5, 5.0),
                # Volume Filter Params
                'volume_filter_lookback': trial.suggest_int('volume_filter_lookback', 10, 50),
                'volume_filter_min_mult': trial.suggest_float('volume_filter_min_mult', 0.5, 1.1),
                # Prediction Filter Params (Ejemplo TF)
                'pred_roc_period': trial.suggest_int('pred_roc_period', 3, 15),
                'pred_roc_threshold_pct': trial.suggest_float('pred_roc_threshold_pct', 0.1, 5.0, log=True),
                'pred_vol_short_period': trial.suggest_int('pred_vol_short_period', 3, 10),
                'pred_vol_long_period': trial.suggest_int('pred_vol_long_period', 15, 50),
                'pred_vol_ratio_max': trial.suggest_float('pred_vol_ratio_max', 1.5, 10.0),
                'pred_vol_ratio_min': trial.suggest_float('pred_vol_ratio_min', 0.1, 0.9),
                'pred_mrev_sma_period': trial.suggest_int('pred_mrev_sma_period', 10, 40),
                'pred_mrev_std_period': trial.suggest_int('pred_mrev_std_period', 10, 40),
                'pred_mrev_zscore_threshold': trial.suggest_float('pred_mrev_zscore_threshold', 1.0, 5.0),
                # Complexity Level
                'strategy_complexity_level': trial.suggest_categorical('strategy_complexity_level', ['no_filters']), # Forzar sin filtros por ahora
            }

            # Combinar parámetros fijos y optimizados
            params = fixed_params_for_objective.copy()
            params.update(opt_params)
            params['trial_db_name'] = db_filename # Pasar nombre de la DB a la estrategia
                        
            logging.debug(f"{log_prefix} Params finales IS (excluye db_name):\n{json.dumps(opt_params, indent=2, default=str)}")


            # --- 3. Configurar y Ejecutar Cerebro IS ---
            # ** USAR runonce=False ** para fiabilidad de notify_trade
            cerebro = bt.Cerebro(stdstats=False, preload=True, runonce=False) # <<< runonce=False !!!
            cerebro.broker.setcash(start_cash)
            cerebro.broker.setcommission(commission=commission)
            if slippage_percentage > 0:
                cerebro.broker.set_slippage_perc(perc=slippage_percentage, slip_open=True, slip_limit=True, slip_match=True, slip_out=False)

            # Crear Data Feed Pandas para IS
            data_is = bt.feeds.PandasData(dataname=dataframe_is.copy(), # Usar copia por si acaso
                                          timeframe=data_timeframe,
                                          compression=data_compression)
            cerebro.adddata(data_is, name="IS_Data") # Darle un nombre opcional
            cerebro.addstrategy(GridBacktraderStrategyV5, **params) # Pasar TODOS los parámetros

            logging.info(f"{log_prefix} Ejecutando cerebro.run() IS (runonce=False)...")
            print(f"INFO PRINT: {log_prefix} Iniciando cerebro.run()... (MODO LENTO: runonce=False)")
            start_run = time.time()
            try:
                # Ejecutar el backtest IS
                results = cerebro.run() # results[0] será la instancia de la estrategia
            except Exception as e_cerebro:
                logging.error(f"{log_prefix} Excepción DURANTE cerebro.run(): {e_cerebro}", exc_info=True)
                # Si cerebro.run falla, la DB puede quedar inconsistente o vacía
                raise RuntimeError(f"Error Cerebro ({type(e_cerebro).__name__})") from e_cerebro
            run_dur = time.time() - start_run
            logging.info(f"{log_prefix} cerebro.run() IS finalizado OK ({run_dur:.2f}s).")
            print(f"INFO PRINT: {log_prefix} cerebro.run() IS OK ({run_dur:.2f}s).")


            # --- 4. Leer DB y Calcular Métricas Manuales ---
            logging.debug(f"{log_prefix} Leyendo resultados de DB: {db_filename}...")
            df_t = pd.DataFrame() # Inicializar DataFrame vacío
            trades = 0 # Inicializar contador de trades

            try: # Try específico para leer la DB
                with sqlite3.connect(db_filename) as conn_r:
                    df_t = pd.read_sql_query("SELECT pnlcomm, barlen FROM trades", conn_r) # Sabemos que esto funcionó

                # ---> SE ELIMINÓ EL BLOQUE GRANDE DE PRINTS DE DEBUG AQUÍ <---

                # ---> SE AÑADIÓ *SOLO* ESTA LÍNEA PRINT <---
                print("DEBUG: >>> Lectura df_t desde DB OK <<<")

        # El código original para calcular 'trades', etc., sigue aquí:
                trades = len(df_t)
                logging.info(f"{log_prefix} Leídos {trades} registros de la DB.")
                print(f"INFO PRINT: {log_prefix} Leídos {len(df_t)} registros de la DB.")

                if trades > 0:
                    print(f"DEBUG: Procesando {trades} trades leídos...") # DEBUG
                    try: # Añadir try específico para conversión y cálculos
                        print("DEBUG: Antes de pd.to_numeric...") # DEBUG
                        pnl_series = pd.to_numeric(df_t['pnlcomm'], errors='coerce').dropna()
                        print(f"DEBUG: Después de pd.to_numeric y dropna. Tipo pnl_series: {type(pnl_series)}") # DEBUG

                        valid_trades = len(pnl_series)
                        print(f"DEBUG: valid_trades = {valid_trades}") # DEBUG
                        logging.debug(f"{log_prefix} Serie PnL post-conversión (len={valid_trades}):\n{pnl_series.head().to_string()}")
                        print(f"INFO PRINT: {log_prefix} Len PnL Series numérico/válido: {valid_trades}")

                        if valid_trades < trades:
                            num_lost = trades - valid_trades
                            loss_p = (num_lost / trades) * 100
                            logging.warning(f"{log_prefix} ¡Se perdieron {num_lost} ({loss_p:.1f}%) registros de PnL en conversión a numérico!")
                            print(f"ADVERTENCIA PRINT: {log_prefix} ¡Se perdieron {num_lost} registros de PnL en conversión!")

                        if valid_trades > 0:
                            print("DEBUG: Calculando métricas...") # DEBUG
                            print("DEBUG:   Antes de pnl = pnl_series.sum()") # DEBUG
                            pnl = pnl_series.sum()
                            print(f"DEBUG:   pnl = {pnl}") # DEBUG

                            print("DEBUG:   Antes de pnl_mean = pnl_series.mean()") # DEBUG
                            pnl_mean = pnl_series.mean()
                            print(f"DEBUG:   pnl_mean = {pnl_mean}") # DEBUG

                            print("DEBUG:   Antes de pnl_std = pnl_series.std()") # DEBUG
                            pnl_std = pnl_series.std()
                            print(f"DEBUG:   pnl_std = {pnl_std}") # DEBUG

                            # Comprobar pnl_std antes de usarlo
                            if pnl_std is None or not np.isfinite(pnl_std) or pnl_std < 1e-9:
                                 print(f"DEBUG:   pnl_std ({pnl_std}) inválido o muy pequeño. Calculando SQN basado en mean.") # DEBUG
                                 sqn = 0.0 if pnl_mean <= 0 else float('inf')
                                 print(f"DEBUG:   sqn (por mean) = {sqn}") # DEBUG
                            else:
                                 print(f"DEBUG:   Antes de sqn = np.sqrt({valid_trades}) * ({pnl_mean} / {pnl_std})") # DEBUG
                                 sqn = np.sqrt(valid_trades) * (pnl_mean / pnl_std)
                                 print(f"DEBUG:   sqn (calculado) = {sqn}") # DEBUG

                            metric_src = "Manual SQLite OK"
                            print("DEBUG: Métricas calculadas OK.") # DEBUG
                            # ---> REEMPLAZA LA LÍNEA logging.info COMPLEJA CON ESTO: <---
                            print(f"DEBUG: Logging metrics step-by-step...")
                            print(f"DEBUG:   log_prefix = {log_prefix}")
                            print(f"DEBUG:   valid_trades = {valid_trades}")
                            print(f"DEBUG:   pnl = {pnl}")
                            print(f"DEBUG:   pnl (formatted): {pnl:.2f}")
                            print(f"DEBUG:   pnl_mean = {pnl_mean}")
                            print(f"DEBUG:   pnl_mean (formatted): {pnl_mean:.2f}")
                            print(f"DEBUG:   pnl_std = {pnl_std}")
                            print(f"DEBUG:   pnl_std (formatted): {pnl_std:.2f}")
                            print(f"DEBUG:   sqn = {sqn}")
                            sqn_str = f"{sqn:.4f}" if np.isfinite(sqn) else str(sqn) # Format separately
                            print(f"DEBUG:   sqn (formatted): {sqn_str}")
                            print(f"DEBUG: Attempting final log message...")
                            # Loguear individualmente para aislar el error:
                            logging.info(f"{log_prefix} Métricas IS - Stage 1: T_valid={valid_trades}")
                            logging.info(f"{log_prefix} Métricas IS - Stage 2: PnL={pnl:.2f}, Mean={pnl_mean:.2f}")
                            logging.info(f"{log_prefix} Métricas IS - Stage 3: Std={pnl_std:.2f}")
                            logging.info(f"{log_prefix} Métricas IS - Stage 4: SQN={sqn_str}")
                            # ---> FIN DEL REEMPLAZO <---


                        else: # if valid_trades > 0:
                            logging.warning(f"{log_prefix} No quedaron trades válidos después de limpiar NaNs. Métricas NaN/0.")
                            pnl,sqn=np.nan,np.nan; metric_src="Err Datos DB (All NaN after dropna)"; score=1.4e11 # Penalización Alta

                    except ValueError as e_calc_ve:
                        logging.error(f"{log_prefix} >>> ValueError DURANTE cálculos PnL/SQN: {e_calc_ve}", exc_info=True)
                        print(f"ERROR PRINT: {log_prefix} >>> ValueError DURANTE cálculos PnL/SQN: {e_calc_ve}")
                        pnl, sqn = np.nan, np.nan
                        metric_src = f"Err Calculo DB (ValueError)"
                        score = 1.6e11 # Penalización diferente para este error
                        raise e_calc_ve # Relanzar para que lo capture el except externo si es necesario o para detener

                    except Exception as e_calc_other:
                        logging.error(f"{log_prefix} >>> Error inesperado DURANTE cálculos PnL/SQN: {e_calc_other}", exc_info=True)
                        print(f"ERROR PRINT: {log_prefix} >>> Error inesperado cálculos PnL/SQN: {e_calc_other}")
                        pnl, sqn = np.nan, np.nan
                        metric_src = f"Err Calculo DB ({type(e_calc_other).__name__})"
                        score = 1.7e11 # Penalización diferente
                        raise e_calc_other # Relanzar

                else: # if trades > 0:
                     # Si la tabla estaba vacía
                    pnl=0.0; sqn=0.0; metric_src="Manual SQLite (0 Trades)";
                    logging.info(f"{log_prefix} No se guardaron trades válidos en DB.")
            except Exception as e_read:
                # Capturar error si falla la lectura de la DB O EL PROCESAMIENTO POSTERIOR
                logging.error(f"{log_prefix} Error durante lectura/procesamiento DB '{db_filename}': {e_read}", exc_info=True) # Loguear traceback
                print(f"ERROR PRINT: {log_prefix} Error durante lectura/procesamiento DB: {e_read}")
                trades=-1 # Marcar como error
                # Usar un nombre más genérico para la razón si el error no fue solo leyendo
                if isinstance(e_read, ValueError):
                     metric_src=f"Err Procesamiento DB (ValueError)"
                elif isinstance(e_read, sqlite3.Error):
                     metric_src=f"Err Lectura DB ({type(e_read).__name__})"
                else:
                     metric_src=f"Err Procesamiento DB ({type(e_read).__name__})"
                score=1.5e11 # Penalización Muy Alta


            # --- 5. Calcular Score Final (MODIFICADO PARA MAXIMIZAR PNL NETO - DIAGNÓSTICO) ---
            final_score = 1.1e12 # Penalización muy alta por defecto (si algo falla abajo)
            min_t = 5 # Mantener el umbral mínimo de trades requeridos

            # Verificar si hubo error en lectura/procesamiento de DB (trades = -1)
            if trades == -1:
                # 'score' y 'metric_src' ya deberían haber sido asignados en el bloque 'except Exception as e_read:'
                # Simplemente usamos ese score de error predefinido.
                final_score = score # Usar el score asignado en el except (e.g., 1.5e11)
                logging.warning(f"{log_prefix} Usando score de error DB/procesamiento: {final_score}")

            # Verificar si hubo suficientes trades válidos
            elif valid_trades < min_t:
                # Penalización por pocos trades (igual que antes)
                final_score = 1e9 + (min_t - valid_trades) * 1e7
                metric_src = f"Pena (T_valid={valid_trades}<{min_t})" # Actualizar razón
                logging.warning(f"{log_prefix} Score final por pocos trades válidos ({valid_trades} < {min_t}): {final_score}")

            # Si hay suficientes trades, intentar usar PnL Neto
            elif pnl is not None and np.isfinite(pnl):
                # ¡OBJETIVO PRINCIPAL AHORA! Optuna minimiza, así que devolvemos -PnL para maximizar PnL.
                final_score = -pnl
                metric_src = f"Base PnL Neto ({pnl:,.2f})" # Actualizar razón
                logging.info(f"{log_prefix} Score final objetivo = -PnL Neto: {final_score:.2f}")
                # >>> OPCIONAL: Guardar SQN como atributo extra para verlo, aunque no se use para score <<<
                try:
                    sqn_value_for_attr = float(sqn) if np.isfinite(sqn) else ('inf' if np.isinf(sqn) else None)
                    trial.set_user_attr("sqn_manual_info", sqn_value_for_attr)
                except Exception as e_sqn_attr:
                    logging.warning(f"{log_prefix} No se pudo guardar sqn_manual_info: {e_sqn_attr}")
                # >>> FIN OPCIONAL <<<

            else:
                # Caso raro: Hubo suficientes trades válidos, pero el PnL Neto calculado es NaN o Inf.
                metric_src = "Pena (PnL Inválido)"
                final_score = 1.2e12 # Usar otra penalización alta distinta
                logging.warning(f"{log_prefix} PnL calculado inválido ({pnl}) a pesar de trades válidos ({valid_trades}). Score de penalización: {final_score}")

            # --- FIN DEL BLOQUE REEMPLAZADO ---

        except Exception as e_obj: # Captura de Excepciones Generales del Trial
            err_info=f"{type(e_obj).__name__}: {str(e_obj)[:150]}" # Guardar info del error
            logging.error(f"{log_prefix} EXCEPCION CATASTRÓFICA en objective_function: {err_info}", exc_info=True)
            final_score=1.1e12 # Score de penalización por error catastrófico
            metric_src=f"Err Inesperado ({err_info})"
            print(f"ERROR PRINT: {log_prefix} EXCEPCION CATASTROFICA: {err_info}")

        # --- 6. Limpieza DB Temporal ---
        if os.path.exists(db_filename):
            try:
                # No cerrar la conexión aquí si se usó 'with', ya está cerrada
                # Intentar borrar el archivo
                os.remove(db_filename)
                logging.debug(f"{log_prefix} DB Temporal '{db_filename}' borrada.")
            except Exception as e_del_final:
                 logging.warning(f"{log_prefix} Warn: No se pudo borrar DB temp final '{db_filename}': {e_del_final}")

        # --- 7. Finalización Trial (Guardar Atributos y Devolver Score) ---
        # Asegurar que el score no sea infinito para Optuna
        final_score = float(final_score) if np.isfinite(final_score) else 1.9e12 # Usar un valor diferente si ya era 1.1e12 o 1.5e11 etc.
        # Añadir tag si se ajustó el score final por no ser finito
        if not np.isfinite(score) and 'Ajustado Max' not in metric_src:
             metric_src += "(Ajustado Max)"

        try:
            # Guardar atributos importantes en el trial de Optuna para análisis posterior
            trial.set_user_attr("score_reason", metric_src)
            trial.set_user_attr("sqn_manual", float(sqn) if np.isfinite(sqn) else ('inf' if np.isinf(sqn) else None))
            trial.set_user_attr("pnl_net_manual", float(pnl) if np.isfinite(pnl) else None)
            trial.set_user_attr("trades_db", int(trades) if trades>=0 else -1) # Trades leídos de DB
            trial.set_user_attr("trades_valid_pnl", int(valid_trades) if trades > 0 else 0) # Trades con PnL finito
            trial.set_user_attr("params_optuna_used", json.loads(json.dumps(opt_params, default=str))) # Guardar solo params optimizados
            if err_info: # Guardar info del error si hubo excepción catastrófica
                trial.set_user_attr("error_info", err_info)
        except Exception as e_attr:
            logging.warning(f"{log_prefix} Error guardando user_attrs: {e_attr}")

        logging.info(f"{log_prefix} {'='*10} FIN Eval Trial #{trial.number}: Score={final_score:.6f} (Razón: {metric_src}) {(err_info if err_info else '')} {'='*10}")
        print(f"INFO PRINT: {log_prefix} Fin Trial #{trial.number}. Score={final_score:.4f}. Razón: {metric_src}")
        return final_score
    # --- Fin objective_function ---


    # --- Ejecución de la Optimización Optuna ---
    study=None; opt_dur=0.0; best_params=None; best_score=float('inf')
    try:
        start_opt=time.time()
        sampler = optuna.samplers.TPESampler(seed=R_SEED) # Sampler por defecto con semilla
        study_name = f"GridOpt_SQLite_v5_{current_run_timestamp.strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(direction='minimize', sampler=sampler, study_name=study_name)

        logging.info(f"Estudio Optuna creado: '{study_name}'")
        logging.info(f"Iniciando study.optimize(n_trials={N_TRIALS}, n_jobs={N_JOBS})...")
        print(f"\n--- [Bloque Principal] Iniciando Optimizacion ({N_JOBS} jobs)... ---")

        # Ejecutar la optimización
        study.optimize(objective_function, n_trials=N_TRIALS, n_jobs=N_JOBS)

        opt_dur = time.time() - start_opt
        logging.info(f"\nOptimización finalizada OK ({opt_dur:.2f}s).")
        print(f"--- Optuna OK ({opt_dur:.2f}s) ---")

        # Procesar mejor resultado si existe
        if study and study.best_trial:
            best_trial = study.best_trial
            best_score = best_trial.value
            best_params = best_trial.params # Solo los parámetros optimizados
            logging.info(f"Mejor trial encontrado: #{best_trial.number}, Score IS(min): {best_score:.6f}")
            logging.info(f"  Mejores Params (Optuna):\n{json.dumps(best_params, indent=2)}")
            # Extraer métricas manuales guardadas para loguear
            sqn_m=best_trial.user_attrs.get('sqn_manual', None)
            pnl_m=best_trial.user_attrs.get('pnl_net_manual', None)
            trd_db=best_trial.user_attrs.get('trades_db', 'N/A')
            trd_val=best_trial.user_attrs.get('trades_valid_pnl', 'N/A')
            reason=best_trial.user_attrs.get('score_reason', 'N/A')
            err=best_trial.user_attrs.get('error_info', None)

            logging.info(f"  Mets IS (Manual): SQN={sqn_m}, PnL={pnl_m}, Trades DB={trd_db}, Trades Val PnL={trd_val}")
            logging.info(f"  Score Razón: {reason}")
            if err: logging.info(f"  Error Info: {err}")
        else:
            logging.warning("Optuna finalizó pero no encontró 'best_trial'.")
            print("--- Warn: Optuna sin best_trial. ---")
            best_params = None # Asegurar que no hay best_params

    except KeyboardInterrupt:
        opt_dur = time.time() - (locals().get('start_opt', time.time()))
        logging.warning(f"\nOptimización INTERRUMPIDA por usuario ({opt_dur:.2f}s).")
        print("\n--- Optimizacion Interrumpida ---")
        # Mantener el estudio actual si existe para análisis parcial
        best_params = study.best_trial.params if study and study.best_trial else None
        best_score = study.best_trial.value if study and study.best_trial else float('inf')

    except Exception as e_opt:
        opt_dur = time.time() - (locals().get('start_opt', time.time()))
        logging.error(f"\n¡ERROR FATAL DURANTE OPTUNA! ({opt_dur:.2f}s): {e_opt}", exc_info=True)
        print(f"\n--- ERROR FATAL Optuna: {e_opt} ---", file=sys.stderr)
        study=None # Invalidar estudio
        best_params=None # No hay best_params


    # ======================================================
    # PARTE D: Proc Resultados IS y Validación OOS
    # ======================================================
    logging.info("\n"+"="*40+"\n--- BLOQUE PRINCIPAL: Proc IS / Valid OOS ---\n"+"="*40)
    print("\n--- [Bloque Principal] Proc IS / Valid OOS ---")

    final_params_oos = None # Parámetros finales para OOS
    if best_params: # Solo proceder si Optuna encontró un mejor conjunto de params
        print(f"  > Mejor Score IS encontrado (min): {best_score:.6f}")
        # Extraer métricas de user_attrs para reporte
        sqn_is, reason_is, trades_db_is, trades_val_is, pnl_is, err_is = None, 'N/A', 'N/A', 'N/A', None, None
        if study and study.best_trial:
            best_trial_report = study.best_trial
            sqn_is = best_trial_report.user_attrs.get('sqn_manual', None)
            pnl_is = best_trial_report.user_attrs.get('pnl_net_manual', None)
            trades_db_is = best_trial_report.user_attrs.get('trades_db', 'N/A')
            trades_val_is = best_trial_report.user_attrs.get('trades_valid_pnl', 'N/A')
            reason_is = best_trial_report.user_attrs.get('score_reason', 'N/A')
            err_is = best_trial_report.user_attrs.get('error_info', None)
        else:
            print("  > Warn: No se pudieron obtener detalles del best_trial.")

        print(f"  > Razón Score IS : {reason_is}")
        if err_is: print(f"  > Error IS Info  : {err_is}")

        # Formatear métricas para impresión
        sqn_f = 'N/A' if sqn_is is None else ('Inf' if sqn_is == 'inf' else f'{sqn_is:.4f}')
        pnl_f = 'N/A' if pnl_is is None else f'${pnl_is:,.2f}'
        print(f"  > SQN Man IS     : {sqn_f}")
        print(f"  > PnL Man IS     : {pnl_f}")
        print(f"  > Trades DB/Valid: {trades_db_is}/{trades_val_is}")

        # Preparar parámetros para la ejecución OOS
        final_params_oos = fixed_params_for_objective.copy()
        final_params_oos.update(best_params) # Añadir los params optimizados
        print("\n=== Params Optimizados IS (para OOS) ===")
        # Imprimir lista ordenada de params optimizados
        [print(f"    - {k:<30}: {v}") for k,v in sorted(best_params.items())]

    else:
        logging.error("No se encontraron 'best_params' de Optuna. Omitiendo validación OOS.")
        print("\n--- No hay parámetros optimizados para ejecutar OOS. ---")


    # --- Validación Out-of-Sample (OOS) ---
    oos_mets = {'ran': False} # Diccionario para métricas OOS
    oos_dur = 0.0
    oos_cerebro = None
    can_oos = (final_params_oos is not None) and (dataframe_oos is not None and not dataframe_oos.empty)

    if can_oos:
        logging.info("\n"+"="*40+"\n--- Iniciando Validación OOS ---\n"+"="*40)
        print("\n--- [Bloque Principal] Validación OOS ---")
        # Loguear y mostrar params completos usados para OOS
        logging.info(f"Params OOS Completos:\n{json.dumps(final_params_oos, indent=2, default=str)}")
        print("  Params Finales para OOS:")
        [print(f"    - {k:<30}: {v} {'(opt)' if k in best_params else '(fijo)'}") for k,v in sorted(final_params_oos.items())]

        try: # Try para la ejecución OOS completa
            # Configurar Cerebro para OOS (con stdstats y analyzers)
            oos_cerebro = bt.Cerebro(stdstats=True, preload=True, runonce=True) # Usar runonce=True para OOS es más rápido
            oos_cerebro.broker.setcash(start_cash)
            oos_cerebro.broker.setcommission(commission=commission)
            if slippage_percentage > 0:
                oos_cerebro.broker.set_slippage_perc(perc=slippage_percentage, slip_open=True, slip_limit=True, slip_match=True, slip_out=False)

            # Añadir Data Feed OOS
            data_oos_feed = bt.feeds.PandasData(dataname=dataframe_oos.copy(),
                                                timeframe=data_timeframe,
                                                compression=data_compression,
                                                name="OOS_Data") # Nombre para logs
            oos_cerebro.adddata(data_oos_feed)

            # Añadir Analyzers estándar
            analyzers = [('ta', bt.analyzers.TradeAnalyzer), ('sqn', bt.analyzers.SQN),
                         ('dd', bt.analyzers.DrawDown), ('ret', bt.analyzers.Returns),
                         ('ann', bt.analyzers.AnnualReturn)]
            [oos_cerebro.addanalyzer(a, _name=n) for n,a in analyzers]

            # Añadir la Estrategia con los params OOS (asegurarse que trial_db_name es None)
            params_oos_strat = final_params_oos.copy()
            params_oos_strat['trial_db_name'] = None # MUY IMPORTANTE para que no intente escribir DB
            oos_cerebro.addstrategy(GridBacktraderStrategyV5, **params_oos_strat)

            print("--- Ejecutando Backtest OOS... ---")
            t_oos = time.time()
            res_oos = oos_cerebro.run() # Ejecutar OOS
            oos_dur = time.time() - t_oos
            print(f"--- Backtest OOS OK ({oos_dur:.2f}s) ---")
            logging.info(f"OOS OK ({oos_dur:.2f}s).")

            # --- Procesar Resultados OOS ---
            logging.info("--- Inspeccionando resultados OOS ---")
            res_ok = False
            try: # Try para procesar resultados OOS
                if isinstance(res_oos, list) and len(res_oos) > 0:
                    strat_oos = res_oos[0] # Obtener la instancia de la estrategia ejecutada
                    logging.info(f"Tipo res_oos[0]: {type(strat_oos)}")
                    if strat_oos is not None:
                        logging.info("Instancia OOS OK. Extrayendo métricas...")
                        oos_mets['ran'] = True
                        oos_mets['start_cash'] = start_cash
                        oos_mets['final_value'] = oos_cerebro.broker.getvalue()
                        oos_mets['pnl_net'] = oos_mets['final_value'] - oos_mets['start_cash']
                        oos_mets['return_pct'] = (oos_mets['pnl_net'] / oos_mets['start_cash'] * 100.0) if abs(start_cash) > 1e-9 else 0.0

                        # Extraer análisis de los analyzers
                        a = strat_oos.analyzers
                        ta = a.ta.get_analysis() if hasattr(a,'ta') else {}
                        sq = a.sqn.get_analysis() if hasattr(a,'sqn') else {}
                        dd = a.dd.get_analysis() if hasattr(a,'dd') else {}
                        ar = a.ann.get_analysis() if hasattr(a,'ann') else {}

                        # Métricas clave
                        oos_mets['trades'] = ta.get('total', {}).get('closed', 0)
                        if oos_mets['trades'] > 0:
                             wt = ta.get('won', {}).get('total', 0)
                             oos_mets['win_rate'] = (wt / oos_mets['trades'] * 100.0)
                             pw = float(ta.get('won', {}).get('pnl', {}).get('total', 0.0))
                             pl_abs = abs(float(ta.get('lost', {}).get('pnl', {}).get('total', 0.0)))
                             oos_mets['pf'] = (pw / pl_abs) if pl_abs > 1e-9 else (float('inf') if pw > 1e-9 else np.nan)
                        else:
                             oos_mets['win_rate'] = np.nan; oos_mets['pf'] = np.nan

                        sq_r = sq.get('sqn')
                        oos_mets['sqn'] = float(sq_r) if sq_r is not None and np.isfinite(sq_r) else np.nan
                        oos_mets['mdd'] = dd.get('max', {}).get('drawdown', np.nan)
                        oos_mets['mdd_len'] = dd.get('max', {}).get('len', np.nan)
                        # Extraer Retorno Anualizado (puede estar anidado)
                        ar_r = next(iter(ar.values()), None) if isinstance(ar, dict) else None
                        oos_mets['ret_ann'] = float(ar_r * 100.0) if ar_r is not None and np.isfinite(ar_r) else np.nan

                        logging.info(f"Resultados OOS extraídos: {json.dumps(oos_mets, default=str)}")

                        # --- Impresión OOS ---
                        print("\n"+"="*45+"\n--- Resultados Detallados Validación OOS ---\n"+"="*45)
                        # Helper local para formatear métricas de forma segura
                        def fmt_metric(val, fmt_str=',.2f', na_val='N/A'):
                             return na_val if val is None or (isinstance(val, float) and np.isnan(val)) else \
                                    ('Inf' if isinstance(val, float) and np.isinf(val) else \
                                    (format(val, fmt_str) if isinstance(val, (int, float)) else str(val)))

                        print(f"Capital Inicial          : ${fmt_metric(oos_mets.get('start_cash'))}")
                        print(f"Capital Final            : ${fmt_metric(oos_mets.get('final_value'))}")
                        print(f"PnL Neto Total           : ${fmt_metric(oos_mets.get('pnl_net'))}")
                        print(f"Retorno Total (%)        : {fmt_metric(oos_mets.get('return_pct'), '.2f')}%")
                        print("-"*45)
                        print(f"Trades Cerrados          : {oos_mets.get('trades', 'N/A')}")
                        print(f"Win Rate (%)             : {fmt_metric(oos_mets.get('win_rate'), '.2f')}%")
                        print(f"Profit Factor            : {fmt_metric(oos_mets.get('pf'), '.2f')}")
                        print(f"SQN (Backtrader)         : {fmt_metric(oos_mets.get('sqn'), '.2f')}")
                        print(f"Max Drawdown (%)         : {fmt_metric(oos_mets.get('mdd'), '.2f')}%")
                        print(f"Max Drawdown Duracion    : {fmt_metric(oos_mets.get('mdd_len'), ',.0f')} barras")
                        print(f"Retorno Anualizado (%)   : {fmt_metric(oos_mets.get('ret_ann'), '.2f')}%")
                        print("="*45)
                        res_ok = True
                    else:
                         logging.error("res_oos[0] (instancia estrategia) es None.")
                         oos_mets['error'] = 'OOS strategy instance is None'
                else:
                    logging.error(f"Resultado de oos_cerebro.run() inválido/vacío ({type(res_oos)}).")
                    oos_mets['error'] = f'OOS run invalid result type: {type(res_oos)}'
            except Exception as e_proc_oos:
                 logging.error(f"¡ERROR procesando resultados OOS!: {e_proc_oos}", exc_info=True)
                 oos_mets['error'] = f"Unexpected error processing OOS results: {e_proc_oos}"

            logging.info("--- Fin inspección OOS ---")
            if not res_ok:
                 print(f"\n--- ERROR procesando resultados OOS: {oos_mets.get('error', 'Desconocido')} ---")
                 print(f"ERROR PRINT: Falla proc OOS: {oos_mets.get('error', 'Desconocido')}")

        except Exception as e_oos_run:
            logging.error(f"¡ERROR FATAL durante ejecución OOS!: {e_oos_run}", exc_info=True)
            print(f"\n--- ERROR CRÍTICO OOS: {e_oos_run} ---", file=sys.stderr)
            print(f"ERROR PRINT: Falla OOS: {e_oos_run}")
            oos_mets={'ran': False, 'error': f"OOS Run Error: {e_oos_run}"}
    else:
        logging.warning("Validación OOS omitida (no hay best_params o datos OOS).")
        print("\n--- VALIDACIÓN OOS OMITIDA ---")


    # ==========================================
    # PARTE E: Análisis Sensibilidad (Comentado)
    # ==========================================
    logging.info("\n"+"="*40+"\n--- BLOQUE PRINCIPAL: Análisis Sensibilidad IS ---\n"+"="*40)
    print("\n--- [Bloque Principal] Análisis Sensibilidad ---")
    # Aquí iría la lógica para el análisis de sensibilidad si se implementara
    print("--- Análisis sensibilidad OMITIDO. ---")


    # ==========================================
    # PARTE F: Guardar Resumen y Detalles
    # ==========================================
    # Re-importar bt por si acaso hubo problemas antes (paranoia)
    try: import backtrader as bt; logging.debug("bt ref ok.")
    except ImportError: logging.error("Fallo re-importando bt en guardado.")

    logging.info("\n"+"="*40+"\n--- BLOQUE PRINCIPAL: Guardado Resultados ---\n"+"="*40)
    print(f"\n--- [Bloque Principal] Guardando Resultados ---")

    # Nombres de archivo basados en timestamp
    ts_suffix = current_run_timestamp.strftime('%Y%m%d_%H%M%S')
    summary_file=f"resumen_opt_{ts_suffix}.txt"
    details_file=f"all_runs_details_OPTUNA_{ts_suffix}.txt"

    # --- Guardar Resumen TXT ---
    logging.info(f"Guardando resumen: '{summary_file}'")
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            # Encabezado
            try:
                ts_str=current_run_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z') if _timezone_info_available and hasattr(current_run_timestamp, 'tzinfo') and current_run_timestamp.tzinfo is not None else current_run_timestamp.strftime('%Y-%m-%d %H:%M:%S (naive)')
            except Exception: ts_str=current_run_timestamp.strftime('%Y-%m-%d %H:%M:%S') # Fallback simple
            f.write(f"--- Resumen Optuna (vD+DX v5 - runonce=False) ---\n")
            f.write(f"Generado: {ts_str}\n")
            f.write(f"UTC: {current_run_timestamp.astimezone(datetime.timezone.utc).isoformat(timespec='seconds')}\n")
            f.write(f"Archivo Datos: {csv_filepath or 'N/A'}\n")

            # Config General
            f.write("\n=== Config General ===\n")
            f.write(f" Capital Inicial: ${start_cash:,.2f}\n")
            f.write(f" Comisión: {commission*100:.3f}%\n")
            f.write(f" Slippage: {slippage_percentage*100:.3f}%\n")
            f.write(f" Timeframe: {bt.TimeFrame.getname(data_timeframe, data_compression)}/{data_compression}\n")
            f.write(f" Split IS/OOS: {split_ratio*100:.0f}% / {100-split_ratio*100:.0f}%\n")
            if dataframe_is is not None: f.write(f" Rango IS: {dataframe_is.index.min()} -> {dataframe_is.index.max()} ({len(dataframe_is)} filas)\n")
            if dataframe_oos is not None: f.write(f" Rango OOS: {dataframe_oos.index.min()} -> {dataframe_oos.index.max()} ({len(dataframe_oos)} filas)\n")

            # Config Optuna
            f.write("\n=== Config Optuna ===\n")
            n_comp = len(study.trials) if study else 0
            f.write(f" Trials Req/Comp: {N_TRIALS}/{n_comp}\n")
            f.write(f" Jobs Paralelos: {N_JOBS}\n")
            f.write(f" Seed Sampler: {R_SEED}\n")
            f.write(f" Duración Opt: {opt_dur:.2f}s\n")
            f.write(f" Modo Ejecución IS: {'Bar-by-Bar (runonce=False)' if 'cerebro' in locals() and not getattr(cerebro, 'p', {}).get('runonce', True) else 'Vectorized (runonce=True)'}\n") # Indicar modo usado

            # Mejor Resultado IS
            f.write("\n=== Mejor Resultado IS ===\n")
            if best_params and study and study.best_trial:
                # Re-extraer por si acaso
                bt = study.best_trial
                sqn_m = bt.user_attrs.get('sqn_manual', None)
                pnl_m = bt.user_attrs.get('pnl_net_manual', None)
                trd_db = bt.user_attrs.get('trades_db', 'N/A')
                trd_val = bt.user_attrs.get('trades_valid_pnl', 'N/A')
                reason = bt.user_attrs.get('score_reason', 'N/A')
                err_i = bt.user_attrs.get('error_info', None)
                # Formateo seguro
                sqn_f = 'N/A' if sqn_m is None else ('Inf' if sqn_m == 'inf' else f'{sqn_m:.4f}')
                pnl_f = 'N/A' if pnl_m is None else f'${pnl_m:,.2f}'

                f.write(f" Trial #: {bt.number}\n")
                f.write(f" Score (min): {best_score:.6f}\n")
                f.write(f" Razón Score: {reason}\n")
                if err_i: f.write(f" Error Info IS: {err_i}\n")
                f.write(f" SQN Manual: {sqn_f}\n")
                f.write(f" PnL Manual: {pnl_f}\n")
                f.write(f" Trades DB/Valid: {trd_db} / {trd_val}\n")
                f.write("\n Parámetros Optimizados:\n")
                [f.write(f"  - {k:<28}: {v}\n") for k,v in sorted(best_params.items())]
            else:
                f.write(" (No disponible o ejecución fallida)\n")

            # Resultados OOS
            f.write("\n=== Resultados OOS ===\n")
            if oos_mets.get('ran'):
                # Usar helper local de nuevo
                def fmt(v, s=',.2f', n='N/A'): return n if v is None or (isinstance(v, float) and np.isnan(v)) else ('Inf' if isinstance(v, float) and np.isinf(v) else (format(v,s) if isinstance(v,(int,float)) else str(v)))
                f.write(f" Capital Final: ${fmt(oos_mets.get('final_value'))}\n")
                f.write(f" PnL Neto: ${fmt(oos_mets.get('pnl_net'))}\n")
                f.write(f" Retorno %: {fmt(oos_mets.get('return_pct'), '.2f')}%\n")
                f.write(f" Trades: {oos_mets.get('trades', 'N/A')}\n")
                f.write(f" Win Rate %: {fmt(oos_mets.get('win_rate'), '.2f')}%\n")
                f.write(f" Profit Factor: {fmt(oos_mets.get('pf'), '.2f')}\n")
                f.write(f" SQN (BT): {fmt(oos_mets.get('sqn'), '.2f')}\n")
                f.write(f" Max Drawdown %: {fmt(oos_mets.get('mdd'), '.2f')}%\n")
                f.write(f" Max DD Dur (b): {fmt(oos_mets.get('mdd_len'), ',.0f')}\n")
                f.write(f" Ret Ann %: {fmt(oos_mets.get('ret_ann'), '.2f')}%\n")
                f.write(f" Duración OOS: {oos_dur:.2f}s\n")
            elif oos_mets.get('error'):
                 f.write(f" Error OOS: {oos_mets['error']}\n")
            else:
                 f.write(" (No ejecutado / Omitido / Falló)\n")

            # Sensibilidad (placeholder)
            f.write("\n=== Sensibilidad ===\n")
            f.write(" (Omitido)\n")

            f.write("\n--- Fin Resumen ---\n")

        logging.info(f"Resumen OK: '{summary_file}'.")
        print(f"--- Resumen OK: '{summary_file}' ---")
    except Exception as e_save_sum:
        logging.error(f"Error guardando resumen txt: {e_save_sum}", exc_info=True)
        print(f"--- ERROR guardando resumen: {e_save_sum} ---")


    # --- Guardar Detalles Completos Optuna ---
    logging.info(f"Guardando detalles Optuna: '{details_file}'")
    try:
        with open(details_file, 'w', encoding='utf-8') as f_all:
            f_all.write(f"--- Detalles Completos Optuna (vD+DX v5 - runonce=False) ---\n")
            f_all.write(f"Timestamp Ejecución: {current_run_timestamp.isoformat()}\n")
            f_all.write(f"Archivo Datos: {csv_filepath or 'N/A'}\n")
            f_all.write(f"Trials Solicitados: {N_TRIALS}\n")

            if study: # Solo si el estudio existe
                f_all.write(f"Nombre Estudio: {study.study_name}\n")
                f_all.write(f"Trials Completados/Registrados: {len(study.trials)}\n")
                f_all.write(f"\nParámetros Fijos:\n{json.dumps(fixed_params_for_objective, indent=2)}\n")
                f_all.write("\n"+"="*50+"\nDETALLES POR TRIAL:\n"+"="*50+"\n")

                # Iterar sobre todos los trials registrados en el estudio
                for t in study.trials:
                    f_all.write(f"\n--- Trial #{t.number} ---\n")
                    # Estado del trial
                    st = t.state.name if hasattr(t.state, 'name') else str(t.state)
                    f_all.write(f"Estado: {st}\n")
                    # Valor/Score obtenido
                    val = f"{t.value:.6f}" if t.value is not None else "N/A"
                    f_all.write(f"Valor (Score): {val}\n")
                    # Timestamps y Duración
                    if t.datetime_start: f_all.write(f"Inicio: {t.datetime_start.isoformat()}\n")
                    if t.datetime_complete: f_all.write(f"Fin: {t.datetime_complete.isoformat()}\n")
                    dur = t.duration.total_seconds() if t.duration else None
                    dur_s = f"{dur:.2f}s" if dur is not None else "N/A"
                    f_all.write(f"Duración: {dur_s}\n")

                    # Parámetros optimizados usados en este trial
                    f_all.write("\n Parámetros Optimizados Sugeridos:\n")
                    p_opt = t.params if t.params else {}
                    if p_opt: [f_all.write(f"  - {k:<28}: {v}\n") for k,v in sorted(p_opt.items())]
                    else: f_all.write("  (N/A)\n")

                    # Atributos de usuario guardados (métricas, razón, error)
                    f_all.write("\n Atributos de Usuario:\n")
                    u_attrs = t.user_attrs if t.user_attrs else {}
                    if u_attrs:
                        # Excluir la repetición de parámetros si se guardó allí
                        [f_all.write(f"  - {k:<20}: {v}\n") for k,v in sorted(u_attrs.items()) if k != 'params_optuna_used']
                    else: f_all.write("  (N/A)\n")
            else:
                f_all.write("\nEstudio Optuna no disponible (posiblemente por error fatal).\n")

            f_all.write("\n--- Fin Detalles Completos ---\n")

        logging.info(f"Detalles Optuna OK: '{details_file}'.")
        print(f"--- Detalles Optuna OK: '{details_file}' ---")
    except Exception as e_save_det:
        logging.error(f"Error guardando detalles Optuna txt: {e_save_det}", exc_info=True)
        print(f"--- ERROR guardando detalles: {e_save_det} ---")


    # --- Final Script ---
    try:
        # Obtener timestamp final formateado
        fin_ts = datetime.datetime.now(lima_tz if _timezone_info_available else None)
        fin_ts_str = fin_ts.strftime('%Y-%m-%d %H:%M:%S %Z') if _timezone_info_available and hasattr(fin_ts, 'tzinfo') and fin_ts.tzinfo is not None else fin_ts.strftime('%Y-%m-%d %H:%M:%S (naive)')
    except Exception: fin_ts_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # Fallback simple

    logging.info("\n"+"="*60+f"\n--- SCRIPT FINALIZADO ({fin_ts_str}) ---\n"+"="*60)
    print("\n--- SCRIPT FINALIZADO ---")

# --- Fin total del archivo .py ---
# --- Fin Parte 13/13 ---
