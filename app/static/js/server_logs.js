document.addEventListener("DOMContentLoaded", function() {
    const logsEnabledMeta = document.querySelector('meta[name="ui-server-logs"]');
    if (logsEnabledMeta && logsEnabledMeta.content === "false") {
        return;
    }

    const panel = document.getElementById("serverLogsPanel");
    const toggle = document.getElementById("serverLogsToggle");
    const followBtn = document.getElementById("serverLogsFollow");
    const body = document.getElementById("serverLogsBody");
    const content = document.getElementById("serverLogsContent");
    const statusLed = document.getElementById("serverLogsStatus");

    if (!panel || !toggle || !followBtn || !body || !content || !statusLed) {
        return;
    }

    const connectionStatusLabels = {
        connecting: "Connecting to log stream...",
        connected: "Log stream connected — server running",
        disconnected: "Log stream disconnected — server may be stopped",
    };

    const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
    let logsWebSocket = null;
    let reconnectTimer = null;
    let logsDisabled = false;
    let unreadCount = 0;
    let followLatest = true;
    const maxRenderedLines = 1500;

    function setConnectionStatus(state) {
        statusLed.className = `server-logs-status server-logs-status-${state}`;
        const label = connectionStatusLabels[state] || state;
        statusLed.title = label;
        statusLed.setAttribute("aria-label", label);
    }

    function setFollowLatest(isFollowing) {
        followLatest = isFollowing;
        followBtn.classList.toggle("is-active", isFollowing);
        followBtn.setAttribute("aria-pressed", String(isFollowing));
        followBtn.title = isFollowing ? "Following latest logs" : "Not following — scroll freely";
        followBtn.setAttribute(
            "aria-label",
            isFollowing ? "Follow latest logs (on)" : "Follow latest logs (off)"
        );
        if (isFollowing && !body.hidden) {
            scrollToBottom();
        }
    }

    function syncPageLayout(isOpen) {
        document.documentElement.classList.toggle("server-logs-open", isOpen);
        document.body.classList.toggle("server-logs-open", isOpen);
    }

    function ensureContentClearsDock() {
        const dock = document.querySelector(".server-logs-wrapper");
        const settings = document.querySelector(".settings");
        if (!dock || !settings || body.hidden) {
            return;
        }

        const dockTop = dock.getBoundingClientRect().top;
        const settingsBottom = settings.getBoundingClientRect().bottom;
        const overlap = settingsBottom - dockTop + 16;
        if (overlap > 0) {
            window.scrollBy({ top: overlap, behavior: "smooth" });
        }
    }

    function setPanelOpen(isOpen) {
        body.hidden = !isOpen;
        toggle.setAttribute("aria-expanded", String(isOpen));
        panel.classList.toggle("is-open", isOpen);
        syncPageLayout(isOpen);
        if (isOpen) {
            unreadCount = 0;
            panel.classList.remove("server-logs-unread");
            requestAnimationFrame(function() {
                ensureContentClearsDock();
                if (followLatest) {
                    scrollToBottom();
                }
            });
        }
    }

    function scrollToBottom() {
        content.scrollTop = content.scrollHeight;
    }

    function tagClassName(tag) {
        const normalized = String(tag || "APP").toLowerCase();
        if (normalized === "sys") {
            return "server-log-tag-sys";
        }
        if (normalized === "info") {
            return "server-log-tag-info";
        }
        if (normalized === "app") {
            return "server-log-tag-app";
        }
        if (normalized === "err" || normalized === "error" || normalized === "critical") {
            return "server-log-tag-err";
        }
        if (normalized === "warning" || normalized === "warn") {
            return "server-log-tag-warning";
        }
        return "server-log-tag-info";
    }

    function clearPlaceholder() {
        const placeholder = content.querySelector(".server-log-placeholder");
        if (placeholder) {
            placeholder.remove();
        }
    }

    function trimRenderedLines() {
        const lines = content.querySelectorAll(".server-log-line");
        const extra = lines.length - maxRenderedLines;
        if (extra <= 0) {
            return;
        }
        for (let i = 0; i < extra; i += 1) {
            lines[i].remove();
        }
    }

    function appendLogLine(line) {
        if (!line || !line.message) {
            return;
        }

        clearPlaceholder();

        const row = document.createElement("div");
        row.className = "server-log-line";

        const ts = document.createElement("span");
        ts.className = "server-log-ts";
        ts.textContent = line.ts || "--:--:--";

        const tag = document.createElement("span");
        tag.className = `server-log-tag ${tagClassName(line.tag)}`;
        tag.textContent = line.tag || "APP";

        const message = document.createElement("span");
        message.className = "server-log-message";
        message.textContent = line.message;

        row.appendChild(ts);
        row.appendChild(tag);
        row.appendChild(message);
        content.appendChild(row);
        trimRenderedLines();

        if (body.hidden) {
            unreadCount += 1;
            panel.classList.add("server-logs-unread");
        } else if (followLatest) {
            scrollToBottom();
        }
    }

    function handleLogPayload(payload) {
        if (payload.action === "log_stream_disabled") {
            logsDisabled = true;
            setConnectionStatus("disconnected");
            if (logsWebSocket) {
                logsWebSocket.close();
            }
            return;
        }
        if (payload.action === "log_history" && Array.isArray(payload.lines)) {
            content.innerHTML = "";
            payload.lines.forEach(appendLogLine);
            if (followLatest && !body.hidden) {
                requestAnimationFrame(scrollToBottom);
            }
            return;
        }
        if (payload.action === "server_log") {
            appendLogLine(payload);
        }
    }

    function connectLogsWebSocket() {
        if (logsDisabled) {
            return;
        }
        if (logsWebSocket && (logsWebSocket.readyState === WebSocket.OPEN || logsWebSocket.readyState === WebSocket.CONNECTING)) {
            return;
        }

        logsWebSocket = new WebSocket(`${wsProtocol}://${window.location.host}/ws/logs`);
        setConnectionStatus("connecting");

        logsWebSocket.onopen = function() {
            setConnectionStatus("connected");
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
                reconnectTimer = null;
            }
        };

        logsWebSocket.onmessage = function(event) {
            try {
                handleLogPayload(JSON.parse(event.data));
            } catch (error) {
                console.error("Failed to parse server log message:", error);
            }
        };

        logsWebSocket.onclose = function(event) {
            logsWebSocket = null;
            setConnectionStatus("disconnected");
            if (logsDisabled || event.code === 1000 || event.code === 1008) {
                logsDisabled = true;
                return;
            }
            reconnectTimer = setTimeout(connectLogsWebSocket, 3000);
        };

        logsWebSocket.onerror = function() {
            setConnectionStatus("disconnected");
            if (logsWebSocket) {
                logsWebSocket.close();
            }
        };
    }

    toggle.addEventListener("click", function() {
        setPanelOpen(body.hidden);
    });

    followBtn.addEventListener("click", function(event) {
        event.stopPropagation();
        setFollowLatest(!followLatest);
    });

    content.addEventListener("wheel", function(event) {
        event.stopPropagation();
    }, { passive: true });

    setPanelOpen(false);
    setFollowLatest(true);
    setConnectionStatus("connecting");
    connectLogsWebSocket();

    window.addEventListener("resize", function() {
        if (!body.hidden) {
            ensureContentClearsDock();
        }
    });

    window.addEventListener("beforeunload", function() {
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
        }
        if (logsWebSocket) {
            logsWebSocket.close();
        }
    });
});
