import { useRef, type ReactNode } from 'react';
import { motion, useScroll, useSpring, useTransform } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import BearingModel3D, { type BearingModel3DProps } from './BearingModel3D';

type BearingDataProps = Pick<BearingModel3DProps, 'bpfoIntensity' | 'bpfiIntensity' | 'severity'>;

interface HeroIntroProps extends BearingDataProps {
  children?: ReactNode;
}

/**
 * Full-viewport hero with the 3D bearing centered, then on scroll the model
 * shrinks and migrates into a docked corner position where it persists for
 * the rest of the page. The hero section's own height defines the scroll
 * range the migration plays out over; once scrolled past it the transform
 * clamps and the model stays docked, so it remains visible alongside
 * `children` (the rest of the dashboard) without unmounting the WebGL canvas.
 */
export default function HeroIntro({
  children,
  bpfoIntensity = 0,
  bpfiIntensity = 0,
  severity = 0,
}: HeroIntroProps) {
  const heroRef = useRef<HTMLDivElement>(null);

  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ['start start', 'end start'],
  });
  const progress = useSpring(scrollYProgress, { stiffness: 90, damping: 22, mass: 0.6 });

  const scale = useTransform(progress, [0, 1], [1, 0.34]);
  const x = useTransform(progress, [0, 1], ['0vw', '-37vw']);
  const y = useTransform(progress, [0, 1], ['12vh', '-36vh']);
  const heroTextOpacity = useTransform(progress, [0, 0.35], [1, 0]);
  const heroTextY = useTransform(progress, [0, 0.35], [0, -40]);
  const indicatorOpacity = useTransform(progress, [0, 0.12], [1, 0]);
  const glowOpacity = useTransform(progress, [0, 1], [0.5, 0.18]);

  return (
    <div className="relative">
      {/* Pin zone: defines the scroll range the migration plays out over */}
      <div ref={heroRef} className="relative h-[170vh]">
        <div className="sticky top-0 h-screen overflow-hidden flex flex-col items-center justify-center">
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

          <motion.div
            style={{ opacity: indicatorOpacity }}
            className="absolute bottom-10 z-10 flex flex-col items-center gap-1 text-slate-500"
          >
            <span className="font-mono text-xs uppercase tracking-widest">scroll</span>
            <motion.div
              animate={{ y: [0, 8, 0] }}
              transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
            >
              <ChevronDown size={20} />
            </motion.div>
          </motion.div>
        </div>
      </div>

      {/* Persistent fixed 3D layer -- never unmounts, just migrates */}
      <div className="fixed inset-0 z-30 pointer-events-none flex items-center justify-center">
        <motion.div
          style={{
            opacity: glowOpacity,
          }}
          className="absolute w-[70vmin] h-[70vmin] rounded-full bg-healthy/20 blur-[100px] animate-pulse-glow"
        />
        <motion.div
          style={{ scale, x, y }}
          className="pointer-events-auto w-[56vmin] h-[56vmin] max-w-[560px] max-h-[560px]"
        >
          <BearingModel3D
            className="w-full h-full"
            bpfoIntensity={bpfoIntensity}
            bpfiIntensity={bpfiIntensity}
            severity={severity}
          />
        </motion.div>
      </div>

      {/* Rest of the page content, scrolled into normally */}
      <div className="relative z-10">{children}</div>
    </div>
  );
}
