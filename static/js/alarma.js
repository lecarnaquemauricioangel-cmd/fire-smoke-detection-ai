/*
 * Sistema de alarma sonora + voz de VigíaFuego.
 * Sirena generada con Web Audio API (sin archivos externos) y advertencia
 * hablada con Web Speech API (speechSynthesis en español). Compartido por
 * resultado.html, video.html y camara.html para no triplicar la lógica.
 */
(function () {
  let audioCtx = null;
  let silenciado = false;
  let ultimoAvisoCooldown = 0;
  let yaDisparadoAlCargar = false;

  function _obtenerAudioCtx() {
    if (!audioCtx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return null;
      audioCtx = new Ctx();
    }
    return audioCtx;
  }

  function desbloquear() {
    const ctx = _obtenerAudioCtx();
    if (ctx && ctx.state === "suspended") {
      ctx.resume().catch(() => {});
    }
  }

  /**
   * Desbloqueo "fuerte": se llama DENTRO del gesto de clic del botón que
   * envía el formulario (procesar imagen/video, iniciar cámara). Reanuda el
   * AudioContext y reproduce un tono realmente audible por Web Audio API
   * (10ms, volumen 0.01) más una síntesis de voz vacía, para que el
   * navegador registre una reproducción real ligada al gesto del usuario
   * -- no solo la creación del contexto. Mientras el fetch del formulario
   * esté en curso, este MISMO AudioContext sigue "running" (no hay recarga
   * de página de por medio), así que cuando llega la respuesta se puede
   * sonar la sirena/voz sin pedir un segundo clic.
   */
  function desbloquearConTono() {
    const ctx = _obtenerAudioCtx();
    if (!ctx) return;
    const reproducirToneInaudible = () => {
      try {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        gain.gain.value = 0.01;
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + 0.01);
      } catch (err) {
        console.warn("[alarma] No se pudo reproducir el tono de desbloqueo:", err);
      }
    };
    if (ctx.state === "suspended") {
      ctx.resume().then(reproducirToneInaudible).catch(() => {});
    } else {
      reproducirToneInaudible();
    }
    if ("speechSynthesis" in window) {
      try {
        const vacio = new SpeechSynthesisUtterance(" ");
        vacio.volume = 0;
        window.speechSynthesis.speak(vacio);
      } catch (err) {
        /* no crítico: solo es un intento de registrar el permiso de voz */
      }
    }
  }

  function reproducirSirena() {
    const ctx = _obtenerAudioCtx();
    if (!ctx) return;
    try {
      if (ctx.state === "suspended") ctx.resume().catch(() => {});
      const ahora = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sawtooth";
      osc.connect(gain);
      gain.connect(ctx.destination);

      const duracionCiclo = 0.5; // barrido ascendente/descendente tipo sirena
      const ciclos = 3;
      const volumen = 0.18; // moderado: audible pero no ensordecedor en la demo

      gain.gain.setValueAtTime(0, ahora);
      gain.gain.linearRampToValueAtTime(volumen, ahora + 0.05);

      for (let i = 0; i < ciclos; i++) {
        const t0 = ahora + i * duracionCiclo;
        osc.frequency.setValueAtTime(600, t0);
        osc.frequency.linearRampToValueAtTime(1000, t0 + duracionCiclo / 2);
        osc.frequency.linearRampToValueAtTime(600, t0 + duracionCiclo);
      }

      const duracionTotal = ciclos * duracionCiclo;
      gain.gain.setValueAtTime(volumen, ahora + duracionTotal - 0.05);
      gain.gain.linearRampToValueAtTime(0, ahora + duracionTotal);

      osc.start(ahora);
      osc.stop(ahora + duracionTotal + 0.05);
    } catch (err) {
      console.warn("[alarma] No se pudo reproducir la sirena:", err);
    }
  }

  function hablar(mensaje) {
    if (!("speechSynthesis" in window)) return;
    try {
      window.speechSynthesis.cancel(); // evita que se acumulen mensajes encimados
      const utterance = new SpeechSynthesisUtterance(mensaje);
      utterance.lang = "es-ES";
      utterance.rate = 1.0;
      utterance.pitch = 1.05;
      utterance.volume = 1.0;
      window.speechSynthesis.speak(utterance);
    } catch (err) {
      console.warn("[alarma] No se pudo reproducir la voz:", err);
    }
  }

  function mensajeParaDeteccion(hayFuego, hayHumo) {
    if (hayFuego && hayHumo) return "¡Alerta crítica! Fuego y humo detectados simultáneamente.";
    if (hayFuego) return "¡Alerta! Fuego detectado en el área.";
    if (hayHumo) return "¡Alerta! Humo detectado en el área.";
    return null;
  }

  function establecerSilencio(valor) {
    silenciado = !!valor;
    if (silenciado && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }

  function estaSilenciado() {
    return silenciado;
  }

  /** Uso: imagen/video ya renderizados (una sola vez, sin cooldown). */
  function alertar(hayFuego, hayHumo) {
    if (silenciado) return;
    const mensaje = mensajeParaDeteccion(hayFuego, hayHumo);
    if (!mensaje) return;
    reproducirSirena();
    hablar(mensaje);
  }

  let intervaloAlarmaContinua = null;

  /**
   * Uso: reproducción de video con severidad activa. Suena de inmediato y
   * repite sirena+voz cada 'intervaloMs' (5-6s) mientras el video esté en
   * reproducción. Debe pararse explícitamente con detenerAlarmaContinua()
   * (video.onpause/onended) para no seguir sonando con el video detenido.
   */
  function iniciarAlarmaContinua(hayFuego, hayHumo, intervaloMs) {
    detenerAlarmaContinua(); // por si ya había un bucle corriendo, evita duplicarlo
    if (silenciado) return;
    const mensaje = mensajeParaDeteccion(hayFuego, hayHumo);
    if (!mensaje) return;
    reproducirSirena();
    hablar(mensaje);
    intervaloAlarmaContinua = setInterval(() => {
      if (silenciado) return; // el usuario pudo silenciar mientras el bucle corría
      reproducirSirena();
      hablar(mensaje);
    }, intervaloMs || 6000);
  }

  /** Detiene el bucle de alarma continua (si hay uno activo) y corta la voz en curso. */
  function detenerAlarmaContinua() {
    if (intervaloAlarmaContinua) {
      clearInterval(intervaloAlarmaContinua);
      intervaloAlarmaContinua = null;
    }
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  }

  /** Uso: bucle de cámara en vivo — evita saturar con voz/sirena en cada frame. */
  function alertarConCooldown(hayFuego, hayHumo, cooldownMs) {
    if (silenciado) return;
    const mensaje = mensajeParaDeteccion(hayFuego, hayHumo);
    if (!mensaje) return;
    const ahora = Date.now();
    if (ahora - ultimoAvisoCooldown < (cooldownMs || 7000)) return;
    ultimoAvisoCooldown = ahora;
    reproducirSirena();
    hablar(mensaje);
  }

  const ID_BANNER = "bannerActivarAudioVigiaFuego";
  const ID_ESTILO_BANNER = "vigiafuego-pulso-style";

  function _mostrarBannerActivacion() {
    if (document.getElementById(ID_BANNER)) return;

    if (!document.getElementById(ID_ESTILO_BANNER)) {
      const estilo = document.createElement("style");
      estilo.id = ID_ESTILO_BANNER;
      estilo.textContent =
        "@keyframes vigiafuego-pulso {" +
        "0%,100% { transform: translateX(-50%) scale(1); }" +
        "50% { transform: translateX(-50%) scale(1.06); } }";
      document.head.appendChild(estilo);
    }

    const banner = document.createElement("div");
    banner.id = ID_BANNER;
    banner.textContent = "🚨 Haga clic aquí para activar audio de emergencia";
    Object.assign(banner.style, {
      position: "fixed",
      top: "14px",
      left: "50%",
      transform: "translateX(-50%)",
      zIndex: "2147483647",
      background: "#dc2626",
      color: "#fff",
      padding: "10px 20px",
      borderRadius: "9999px",
      fontFamily: "system-ui, -apple-system, sans-serif",
      fontWeight: "700",
      fontSize: "14px",
      lineHeight: "1.2",
      boxShadow: "0 0 0 3px rgba(220,38,38,0.35), 0 8px 24px rgba(0,0,0,0.45)",
      cursor: "pointer",
      userSelect: "none",
      animation: "vigiafuego-pulso 1.2s ease-in-out infinite",
    });
    document.body.appendChild(banner);
  }

  function _ocultarBannerActivacion() {
    const banner = document.getElementById(ID_BANNER);
    if (banner) banner.remove();
  }

  /**
   * Uso: al cargar resultado.html/video.html. Intenta sonar de inmediato;
   * si el navegador bloquea el autoplay (AudioContext queda "suspended"),
   * muestra un banner flotante y queda armado para disparar la sirena+voz
   * en cualquier primer clic, tecla o toque en CUALQUIER parte de la
   * ventana (listener en 'document', no en un botón concreto), sin
   * necesitar recargar la página.
   */
  function alertarAlCargar(hayFuego, hayHumo) {
    const ctx = _obtenerAudioCtx();
    if (!ctx) {
      alertar(hayFuego, hayHumo); // sin Web Audio API: al menos intenta la voz
      return;
    }
    const disparar = () => {
      if (yaDisparadoAlCargar) return;
      yaDisparadoAlCargar = true;
      _ocultarBannerActivacion();
      alertar(hayFuego, hayHumo);
    };

    ctx.resume().then(() => {
      if (ctx.state === "running") {
        disparar();
      } else {
        _mostrarBannerActivacion();
      }
    }).catch(() => _mostrarBannerActivacion());

    const activarConGesto = () => {
      _ocultarBannerActivacion(); // desaparece en cuanto el usuario toca la pantalla
      ctx.resume().then(disparar).catch(() => {});
    };
    ["click", "keydown", "touchstart"].forEach((evt) =>
      document.addEventListener(evt, activarConGesto, { once: true })
    );
  }

  /**
   * Cablea el botón "Desactivar alarma / Silenciar" ↔ "Reactivar alarma".
   * Se comparte entre la carga clásica de página completa (scripts inline
   * de resultado.html/video.html) y la actualización vía fetch (index.html/
   * video.html), ya que un <script> insertado con innerHTML nunca se
   * ejecuta solo -- hay que volver a enganchar el listener explícitamente.
   *
   * 'opciones.continuo' + 'opciones.videoEl' habilitan el modo de video: al
   * reactivar, si el video sigue reproduciéndose, retoma el BUCLE (sirena+
   * voz cada 'opciones.intervaloMs') en vez de un solo aviso puntual; al
   * silenciar, corta ese bucle en seco.
   */
  function inicializarBotonSilenciar(boton, hayFuego, hayHumo, opciones) {
    if (!boton) return;
    const continuo = !!(opciones && opciones.continuo);
    const videoEl = opciones && opciones.videoEl;
    const intervaloMs = (opciones && opciones.intervaloMs) || 6000;

    boton.addEventListener("click", () => {
      const silenciarAhora = !estaSilenciado();
      establecerSilencio(silenciarAhora);
      if (silenciarAhora) {
        if (continuo) detenerAlarmaContinua();
      } else {
        if ("speechSynthesis" in window) window.speechSynthesis.cancel();
        if (continuo && videoEl && !videoEl.paused) {
          iniciarAlarmaContinua(hayFuego, hayHumo, intervaloMs);
        } else {
          alertar(hayFuego, hayHumo);
        }
      }
      boton.innerHTML = silenciarAhora
        ? '<i class="fa-solid fa-volume-high"></i> Reactivar alarma'
        : '<i class="fa-solid fa-volume-xmark"></i> Desactivar alarma / Silenciar';
    });
  }

  window.AlarmaVigiaFuego = {
    desbloquear,
    desbloquearConTono,
    reproducirSirena,
    hablar,
    alertar,
    alertarConCooldown,
    alertarAlCargar,
    iniciarAlarmaContinua,
    detenerAlarmaContinua,
    establecerSilencio,
    estaSilenciado,
    inicializarBotonSilenciar,
  };
})();
