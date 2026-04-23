import { useCallback, useEffect, useMemo, useState } from "react";
import { Routes, Route, NavLink, Navigate } from "react-router-dom";
import {
  Activity,
  BarChart3,
  Clock,
  Code,
  Database,
  Eye,
  FileText,
  Globe,
  Heart,
  KeyRound,
  Menu,
  MessageSquare,
  Package,
  Puzzle,
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
import StatusPage from "@/pages/StatusPage";
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
  { path: "/", labelKey: "status", label: "Status", icon: Activity },
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
    <div className="text-midground font-mondwest bg-black min-h-screen uppercase antialiased">
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

      <div className="lg:flex">
        <aside
          id="app-sidebar"
          aria-label={t.app.navigation}
          className={cn(
            "fixed top-0 left-0 z-50 h-screen w-64",
            "flex flex-col",
            "border-r border-current/20",
            "bg-background-base/95 backdrop-blur-sm",
            "transition-transform duration-200 ease-out",
            mobileOpen ? "translate-x-0" : "-translate-x-full",
            "lg:sticky lg:translate-x-0 lg:shrink-0",
          )}
        >
          <div
            className={cn(
              "flex items-center justify-between gap-2",
              "h-14 shrink-0 px-5",
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
                "lg:hidden inline-flex h-7 w-7 items-center justify-center",
                "text-midground/70 hover:text-midground transition-colors cursor-pointer",
                "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground",
              )}
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <nav
            aria-label={t.app.navigation}
            className="flex-1 overflow-y-auto overflow-x-hidden py-2"
          >
            <ul className="flex flex-col">
              {navItems.map(({ path, label, labelKey, icon: Icon }) => (
                <li key={path}>
                  <NavLink
                    to={path}
                    end={path === "/"}
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
                            className="absolute left-0 top-0 bottom-0 w-px bg-midground"
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

          <div
            className={cn(
              "shrink-0 flex items-center justify-between gap-2",
              "px-3 py-2",
              "border-t border-current/20",
            )}
          >
            <ThemeSwitcher dropUp />
            <LanguageSwitcher />
          </div>

          <div
            className={cn(
              "shrink-0 flex items-center justify-between gap-2",
              "px-5 py-3",
              "border-t border-current/20",
            )}
          >
            <Typography
              mondwest
              className="text-[0.7rem] tracking-[0.12em] opacity-60"
            >
              {t.app.footer.name}
            </Typography>
            <Typography
              mondwest
              className="text-[0.65rem] tracking-[0.15em] text-midground"
              style={{ mixBlendMode: "plus-lighter" }}
            >
              {t.app.footer.org}
            </Typography>
          </div>
        </aside>

        <main
          className={cn(
            "relative z-2 flex-1 min-w-0",
            "px-3 sm:px-6 pb-4 sm:pb-8",
            "pt-16 lg:pt-6",
          )}
        >
          <div className="mx-auto w-full max-w-[1400px]">
            <Routes>
              <Route path="/" element={<StatusPage />} />
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

              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  );
}

interface NavItem {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  labelKey?: string;
  path: string;
}
