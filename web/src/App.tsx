import { useCallback, useEffect, useMemo, useState } from "react";
import { Routes, Route, NavLink, Navigate, useNavigate } from "react-router-dom";
import {
  Activity,
  BarChart3,
  Clock,
  Code,
  Database,
  Download,
  Eye,
  FileText,
  Globe,
  Heart,
  KeyRound,
  Loader2,
  Menu,
  MessageSquare,
  Package,
  Puzzle,
  RotateCw,
  Settings,
  Shield,
  Sparkles,
  Star,
  Terminal,
  Wrench,
  X,
  Zap,
} from "lucide-react";
import { SelectionSwitcher, Typography } from "@nous-research/ui";
import { cn } from "@/lib/utils";
import { Backdrop } from "@/components/Backdrop";
import { SidebarFooter } from "@/components/SidebarFooter";
import { SidebarStatusStrip } from "@/components/SidebarStatusStrip";
import { PageHeaderProvider } from "@/contexts/PageHeaderProvider";
import { useSystemActions } from "@/contexts/useSystemActions";
import type { SystemAction } from "@/contexts/system-actions-context";
import ConfigPage from "@/pages/ConfigPage";
import EnvPage from "@/pages/EnvPage";
import SessionsPage from "@/pages/SessionsPage";
import LogsPage from "@/pages/LogsPage";
import AnalyticsPage from "@/pages/AnalyticsPage";
import CronPage from "@/pages/CronPage";
import SkillsPage from "@/pages/SkillsPage";
import { LanguageSwitcher } from "@/components/LanguageSwitcher";
import { ThemeSwitcher } from "@/components/ThemeSwitcher";
import { useI18n } from "@/i18n";
import { usePlugins } from "@/plugins";
import type { RegisteredPlugin } from "@/plugins";

const BUILTIN_NAV: NavItem[] = [
  {
    path: "/sessions",
    labelKey: "sessions",
    label: "Sessions",
    icon: MessageSquare,
  },
  {
    path: "/analytics",
    labelKey: "analytics",
    label: "Analytics",
    icon: BarChart3,
  },
  { path: "/logs", labelKey: "logs", label: "Logs", icon: FileText },
  { path: "/cron", labelKey: "cron", label: "Cron", icon: Clock },
  { path: "/skills", labelKey: "skills", label: "Skills", icon: Package },
  { path: "/config", labelKey: "config", label: "Config", icon: Settings },
  { path: "/env", labelKey: "keys", label: "Keys", icon: KeyRound },
];

// Plugins can reference any of these by name in their manifest — keeps bundle
// size sane vs. importing the full lucide-react set.
const ICON_MAP: Record<string, React.ComponentType<{ className?: string }>> = {
  Activity,
  BarChart3,
  Clock,
  FileText,
  KeyRound,
  MessageSquare,
  Package,
  Settings,
  Puzzle,
  Sparkles,
  Terminal,
  Globe,
  Database,
  Shield,
  Wrench,
  Zap,
  Heart,
  Star,
  Code,
  Eye,
};

function resolveIcon(
  name: string,
): React.ComponentType<{ className?: string }> {
  return ICON_MAP[name] ?? Puzzle;
}

function buildNavItems(
  builtIn: NavItem[],
  plugins: RegisteredPlugin[],
): NavItem[] {
  const items = [...builtIn];

  for (const { manifest } of plugins) {
    const pluginItem: NavItem = {
      path: manifest.tab.path,
      label: manifest.label,
      icon: resolveIcon(manifest.icon),
    };

    const pos = manifest.tab.position ?? "end";
    if (pos === "end") {
      items.push(pluginItem);
    } else if (pos.startsWith("after:")) {
      const target = "/" + pos.slice(6);
      const idx = items.findIndex((i) => i.path === target);
      items.splice(idx >= 0 ? idx + 1 : items.length, 0, pluginItem);
    } else if (pos.startsWith("before:")) {
      const target = "/" + pos.slice(7);
      const idx = items.findIndex((i) => i.path === target);
      items.splice(idx >= 0 ? idx : items.length, 0, pluginItem);
    } else {
      items.push(pluginItem);
    }
  }

  return items;
}

export default function App() {
  const { t } = useI18n();
  const { plugins } = usePlugins();
  const [mobileOpen, setMobileOpen] = useState(false);

  const closeMobile = useCallback(() => setMobileOpen(false), []);

  const navItems = useMemo(
    () => buildNavItems(BUILTIN_NAV, plugins),
    [plugins],
  );

  const pluginTabMeta = useMemo(
    () =>
      plugins.map((p) => ({
        path: p.manifest.tab.path,
        label: p.manifest.label,
      })),
    [plugins],
  );

  // Close on Escape and lock body scroll while the drawer is open on mobile.
  useEffect(() => {
    if (!mobileOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMobileOpen(false);
    };
    document.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [mobileOpen]);

  // If the user resizes past the lg breakpoint while the drawer is open, drop
  // the mobile state so the scroll lock releases and the persistent sidebar
  // takes over cleanly.
  useEffect(() => {
    const mql = window.matchMedia("(min-width: 1024px)");
    const onChange = (e: MediaQueryListEvent) => {
      if (e.matches) setMobileOpen(false);
    };
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);

  return (
    <div className="flex h-dvh max-h-dvh min-h-0 flex-col overflow-hidden bg-black font-mondwest uppercase text-midground antialiased">
      <SelectionSwitcher />
      <Backdrop />

      {/* Mobile top bar — only visible below the lg breakpoint. */}
      <header
        className={cn(
          "lg:hidden fixed top-0 left-0 right-0 z-40 h-12",
          "flex items-center gap-2 px-3",
          "border-b border-current/20",
          "bg-background-base/90 backdrop-blur-sm",
        )}
      >
        <button
          type="button"
          onClick={() => setMobileOpen(true)}
          aria-label={t.app.openNavigation}
          aria-expanded={mobileOpen}
          aria-controls="app-sidebar"
          className={cn(
            "inline-flex h-8 w-8 items-center justify-center",
            "text-midground/70 hover:text-midground transition-colors cursor-pointer",
            "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground",
          )}
        >
          <Menu className="h-4 w-4" />
        </button>

        <Typography
          className="font-bold text-[0.95rem] leading-[0.95] tracking-[0.05em] text-midground"
          style={{ mixBlendMode: "plus-lighter" }}
        >
          {t.app.brand}
        </Typography>
      </header>

      {/* Scrim behind the mobile drawer. */}
      {mobileOpen && (
        <button
          type="button"
          aria-label={t.app.closeNavigation}
          onClick={closeMobile}
          className={cn(
            "lg:hidden fixed inset-0 z-40",
            "bg-black/60 backdrop-blur-sm cursor-pointer",
          )}
        />
      )}

      <div
        className={cn(
          "relative z-2 flex min-h-0 w-full min-w-0 flex-1 flex-col",
          "pt-16",
          "lg:pl-64 lg:pt-0",
        )}
      >
        <aside
          id="app-sidebar"
          aria-label={t.app.navigation}
          className={cn(
            "fixed top-0 left-0 z-50 flex h-dvh max-h-dvh w-64 min-h-0 flex-col",
            "border-r border-current/20",
            "bg-background-base/95 backdrop-blur-sm",
            "transition-transform duration-200 ease-out",
            mobileOpen ? "translate-x-0" : "-translate-x-full",
            "lg:translate-x-0",
          )}
        >
          <div
            className={cn(
              "flex h-14 shrink-0 items-center justify-between gap-2 px-5",
              "border-b border-current/20",
            )}
          >
            <Typography
              className="font-bold text-[1.125rem] leading-[0.95] tracking-[0.0525rem] text-midground"
              style={{ mixBlendMode: "plus-lighter" }}
            >
              Hermes
              <br />
              Agent
            </Typography>

            <button
              type="button"
              onClick={closeMobile}
              aria-label={t.app.closeNavigation}
              className={cn(
                "inline-flex h-7 w-7 items-center justify-center",
                "text-midground/70 hover:text-midground transition-colors cursor-pointer",
                "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground",
                "lg:hidden",
              )}
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <nav
            aria-label={t.app.navigation}
            className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden py-2"
          >
            <ul className="flex flex-col">
              {navItems.map(({ path, label, labelKey, icon: Icon }) => (
                <li key={path}>
                  <NavLink
                    to={path}
                    end={path === "/sessions"}
                    onClick={closeMobile}
                    className={({ isActive }) =>
                      cn(
                        "group relative flex items-center gap-3",
                        "px-5 py-2.5",
                        "font-mondwest text-[0.8rem] tracking-[0.12em]",
                        "whitespace-nowrap transition-colors cursor-pointer",
                        "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground",
                        isActive
                          ? "text-midground"
                          : "opacity-60 hover:opacity-100",
                      )
                    }
                  >
                    {({ isActive }) => (
                      <>
                        <Icon className="h-3.5 w-3.5 shrink-0" />
                        <span className="truncate">
                          {labelKey
                            ? ((t.app.nav as Record<string, string>)[
                                labelKey
                              ] ?? label)
                            : label}
                        </span>

                        <span
                          aria-hidden
                          className="absolute inset-y-0.5 left-1.5 right-1.5 bg-midground opacity-0 pointer-events-none transition-opacity duration-200 group-hover:opacity-5"
                        />

                        {isActive && (
                          <span
                            aria-hidden
                            className="absolute top-0 bottom-0 left-0 w-px bg-midground"
                            style={{ mixBlendMode: "plus-lighter" }}
                          />
                        )}
                      </>
                    )}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>

          <SidebarSystemActions onNavigate={closeMobile} />

          <div
            className={cn(
              "flex shrink-0 items-center justify-between gap-2",
              "px-3 py-2",
              "border-t border-current/10",
            )}
          >
            <ThemeSwitcher dropUp />
            <LanguageSwitcher />
          </div>

          <SidebarFooter />
        </aside>

        <PageHeaderProvider pluginTabs={pluginTabMeta}>
          <div
            className={cn(
              "min-h-0 w-full min-w-0 flex-1",
              "px-3 pb-4 sm:px-6 sm:pb-8",
              "pt-2 sm:pt-4",
            )}
          >
            <div className="w-full min-w-0">
              <Routes>
                <Route
                  path="/"
                  element={<Navigate to="/sessions" replace />}
                />
                <Route path="/sessions" element={<SessionsPage />} />
                <Route path="/analytics" element={<AnalyticsPage />} />
                <Route path="/logs" element={<LogsPage />} />
                <Route path="/cron" element={<CronPage />} />
                <Route path="/skills" element={<SkillsPage />} />
                <Route path="/config" element={<ConfigPage />} />
                <Route path="/env" element={<EnvPage />} />

                {plugins.map(({ manifest, component: PluginComponent }) => (
                  <Route
                    key={manifest.name}
                    path={manifest.tab.path}
                    element={<PluginComponent />}
                  />
                ))}

                <Route path="*" element={<Navigate to="/sessions" replace />} />
              </Routes>
            </div>
          </div>
        </PageHeaderProvider>
      </div>
    </div>
  );
}

function SidebarSystemActions({ onNavigate }: { onNavigate: () => void }) {
  const { t } = useI18n();
  const navigate = useNavigate();
  const { activeAction, isBusy, isRunning, pendingAction, runAction } =
    useSystemActions();

  const items: SystemActionItem[] = [
    {
      action: "restart",
      icon: RotateCw,
      label: t.status.restartGateway,
      runningLabel: t.status.restartingGateway,
      spin: true,
    },
    {
      action: "update",
      icon: Download,
      label: t.status.updateHermes,
      runningLabel: t.status.updatingHermes,
      spin: false,
    },
  ];

  const handleClick = (action: SystemAction) => {
    if (isBusy) return;
    void runAction(action);
    navigate("/sessions");
    onNavigate();
  };

  return (
    <div
      className={cn(
        "shrink-0 flex flex-col",
        "border-t border-current/10",
        "py-1",
      )}
    >
      <span
        className={cn(
          "px-5 pt-0.5 pb-0.5",
          "font-mondwest text-[0.6rem] tracking-[0.15em] uppercase opacity-30",
        )}
      >
        {t.app.system}
      </span>

      <SidebarStatusStrip />

      <ul className="flex flex-col">
        {items.map(({ action, icon: Icon, label, runningLabel, spin }) => {
          const isPending = pendingAction === action;
          const isActionRunning =
            activeAction === action && isRunning && !isPending;
          const busy = isPending || isActionRunning;
          const displayLabel = isActionRunning ? runningLabel : label;
          const disabled = isBusy && !busy;

          return (
            <li key={action}>
              <button
                type="button"
                onClick={() => handleClick(action)}
                disabled={disabled}
                aria-busy={busy}
                className={cn(
                  "group relative flex w-full items-center gap-3",
                  "px-5 py-1.5",
                  "font-mondwest text-[0.75rem] tracking-[0.1em]",
                  "text-left whitespace-nowrap transition-opacity cursor-pointer",
                  "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground",
                  busy
                    ? "text-midground opacity-100"
                    : "opacity-60 hover:opacity-100",
                  "disabled:cursor-not-allowed disabled:opacity-30",
                )}
              >
                {isPending ? (
                  <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
                ) : (
                  <Icon
                    className={cn(
                      "h-3.5 w-3.5 shrink-0",
                      isActionRunning && spin && "animate-spin",
                      isActionRunning && !spin && "animate-pulse",
                    )}
                  />
                )}

                <span className="truncate">{displayLabel}</span>

                <span
                  aria-hidden
                  className="absolute inset-y-0.5 left-1.5 right-1.5 bg-midground opacity-0 pointer-events-none transition-opacity duration-200 group-hover:opacity-5"
                />

                {busy && (
                  <span
                    aria-hidden
                    className="absolute left-0 top-0 bottom-0 w-px bg-midground"
                    style={{ mixBlendMode: "plus-lighter" }}
                  />
                )}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

interface NavItem {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  labelKey?: string;
  path: string;
}

interface SystemActionItem {
  action: SystemAction;
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  runningLabel: string;
  spin: boolean;
}
