import { useRef, type ReactNode } from 'react';
import { motion, useScroll, useSpring, useTransform } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import BearingModel3D, { type BearingModel3DProps } from './BearingModel3D';

type BearingDataProps = Pick<BearingModel3DProps, 'bpfoIntensity' | 'bpfiIntensity' | 'severity'>;

interface HeroIntroProps extends BearingDataProps {
  children?: ReactNode;
  /** Controls slot (FileScrubber) rendered at the bottom of the hero. */
  heroSlot?: ReactNode;
  /**
   * Stat cards rendered to the right of the bearing in the hero.
   * These fade with the bearing so they're prominent in the hero and ghost when
   * the user scrolls into the dashboard below.
   * On narrow viewports (<lg) they stack below the bearing instead.
   */
  heroStats?: ReactNode;
}

/**
 * Full-viewport hero.
 *
 * Bearing (z-5) never moves or changes size. Only opacity animates:
 *   0 – 25%  of hero scroll  →  opacity 1   (full presence)
 *   25 – 75% of hero scroll  →  fades to 0.12
 *   75 – 100%                →  opacity 0.12 (ambient ghost)
 *
 * Dashboard content is at z-10, so it scrolls above the z-5 bearing.
 */
export default function HeroIntro({
  children,
  heroSlot,
  heroStats,
  bpfoIntensity = 0,
  bpfiIntensity = 0,
  severity = 0,
}: HeroIntroProps) {
  const heroRef = useRef<HTMLDivElement>(null);

  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ['start start', 'end start'],
  });
  const progress = useSpring(scrollYProgress, { stiffness: 100, damping: 24, mass: 0.6 });

  const bearingOpacity   = useTransform(progress, [0.25, 0.75], [1, 0.12]);
  const heroTextOpacity  = useTransform(progress, [0, 0.35], [1, 0]);
  const heroTextY        = useTransform(progress, [0, 0.35], [0, -40]);
  const indicatorOpacity = useTransform(progress, [0, 0.15], [1, 0]);
  const glowOpacity      = useTransform(progress, [0.25, 0.75], [0.45, 0]);

  return (
    <div className="relative">
      <div ref={heroRef} className="relative h-[170vh]">
        <div className="sticky top-0 h-screen overflow-hidden flex flex-col items-center justify-center">
          {/* Title */}
          <motion.div
            style={{ opacity: heroTextOpacity, y: heroTextY }}
            className="absolute top-[5%] z-10 text-center px-4"
          >
            <h1 className="font-sans font-bold text-5xl md:text-7xl tracking-tight text-white">
              Bearing RUL Prediction
            </h1>
            <p className="mt-4 font-mono text-sm md:text-base text-slate-400">
              FrSST {'→'} physics-informed fault matching {'→'} CNN health index {'→'} RUL
            </p>
          </motion.div>

          {/* Scrubber (bottom of hero, above bearing) */}
          {heroSlot && (
            <div className="absolute bottom-[8%] left-4 right-4 z-40">
              {heroSlot}
            </div>
          )}

          <motion.div
            style={{ opacity: indicatorOpacity }}
            className="absolute bottom-3 z-10 flex flex-col items-center gap-1 text-slate-500"
          >
            <motion.div
              animate={{ y: [0, 8, 0] }}
              transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
            >
              <ChevronDown size={18} />
            </motion.div>
          </motion.div>
        </div>
      </div>

      {/* ── Bearing layer ─────────────────────────────────────────────────────
          z-5: below z-10 dashboard content, so charts always render on top.
          Only opacity changes with scroll — position and size are constant.
      ─────────────────────────────────────────────────────────────────────── */}
      <div className="fixed inset-0 z-[5] pointer-events-none flex items-center justify-center">
        {/* Ambient glow */}
        <motion.div
          style={{ opacity: glowOpacity }}
          className="absolute w-[76vmin] h-[76vmin] rounded-full bg-healthy/20 blur-[120px] animate-pulse-glow"
        />

        {/* Bearing + stat cards, side-by-side on lg+, stacked on smaller screens */}
        <motion.div
          style={{ opacity: bearingOpacity, translateY: '3vh' }}
          className="pointer-events-auto flex flex-col lg:flex-row items-center gap-6 px-4"
        >
          {/* 3D bearing */}
          <div className="w-[62vmin] h-[62vmin] shrink-0">
            <BearingModel3D
              className="w-full h-full"
              bpfoIntensity={bpfoIntensity}
              bpfiIntensity={bpfiIntensity}
              severity={severity}
            />
          </div>

          {/* Stat cards column */}
          {heroStats && (
            <div className="flex flex-col gap-3 w-64 shrink-0">
              {heroStats}
            </div>
          )}
        </motion.div>
      </div>

      {/* Dashboard panels — z-10 scrolls above the z-5 bearing */}
      <div className="relative z-10">{children}</div>
    </div>
  );
}
