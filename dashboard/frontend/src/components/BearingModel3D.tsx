import { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import * as THREE from 'three';

const N_ROLLERS = 16;
const OUTER_RADIUS = 2.6;
const INNER_RADIUS = 1.5;
const ROLLER_ORBIT_RADIUS = (OUTER_RADIUS + INNER_RADIUS) / 2;
const ROLLER_SIZE = 0.26;
const ROW_OFFSET = 0.55;

const STEEL_COLOR = '#9aa3ad';
const WARNING_GLOW = '#f59e0b';
const FAILURE_GLOW = '#ef4444';

function lerpColor(a: string, b: string, t: number): THREE.Color {
  const ca = new THREE.Color(a);
  const cb = new THREE.Color(b);
  return ca.lerp(cb, THREE.MathUtils.clamp(t, 0, 1));
}

interface RaceProps {
  radius: number;
  tube: number;
  matchIntensity: number; // 0-1, drives emissive glow
}

function Race({ radius, tube, matchIntensity }: RaceProps) {
  const matRef = useRef<THREE.MeshStandardMaterial>(null);

  useFrame((state) => {
    if (!matRef.current) return;
    const pulse = matchIntensity > 0.02
      ? 0.5 + 0.5 * Math.sin(state.clock.elapsedTime * 4)
      : 0;
    const target = matchIntensity > 0.66
      ? lerpColor(WARNING_GLOW, FAILURE_GLOW, (matchIntensity - 0.66) / 0.34)
      : lerpColor(STEEL_COLOR, WARNING_GLOW, matchIntensity / 0.66);
    matRef.current.emissive.lerp(target, 0.08);
    matRef.current.emissiveIntensity = THREE.MathUtils.lerp(
      matRef.current.emissiveIntensity,
      matchIntensity * (0.6 + 0.6 * pulse),
      0.1,
    );
  });

  return (
    <mesh rotation={[Math.PI / 2, 0, 0]}>
      <torusGeometry args={[radius, tube, 24, 96]} />
      <meshStandardMaterial
        ref={matRef}
        color={STEEL_COLOR}
        metalness={0.92}
        roughness={0.22}
        emissive={STEEL_COLOR}
        emissiveIntensity={0}
      />
    </mesh>
  );
}

function RollerRow({ zOffset }: { zOffset: number }) {
  const positions = useMemo(() => {
    return Array.from({ length: N_ROLLERS }, (_, i) => {
      const angle = (i / N_ROLLERS) * Math.PI * 2;
      return [
        Math.cos(angle) * ROLLER_ORBIT_RADIUS,
        Math.sin(angle) * ROLLER_ORBIT_RADIUS,
        zOffset,
      ] as [number, number, number];
    });
  }, []);

  return (
    <group>
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, zOffset]}>
        <torusGeometry args={[ROLLER_ORBIT_RADIUS, 0.05, 16, 96]} />
        <meshStandardMaterial color="#3f4654" metalness={0.6} roughness={0.5} />
      </mesh>
      {positions.map((pos, i) => (
        <mesh key={i} position={pos} castShadow receiveShadow>
          <sphereGeometry args={[ROLLER_SIZE, 24, 24]} />
          <meshStandardMaterial color="#e2e8f0" metalness={0.95} roughness={0.12} />
        </mesh>
      ))}
    </group>
  );
}

interface BearingRigProps {
  bpfoIntensity?: number;
  bpfiIntensity?: number;
  severity?: number;
  autoRotate?: boolean;
}

function BearingRig({
  bpfoIntensity = 0,
  bpfiIntensity = 0,
  severity = 0,
  autoRotate = true,
}: BearingRigProps) {
  const groupRef = useRef<THREE.Group>(null);
  const innerAssemblyRef = useRef<THREE.Group>(null);

  useFrame((_state, delta) => {
    if (innerAssemblyRef.current && autoRotate) {
      innerAssemblyRef.current.rotation.z += delta * 0.6;
    }
    if (groupRef.current) {
      const jitter = severity * 0.04;
      groupRef.current.position.x = (Math.random() - 0.5) * jitter;
      groupRef.current.position.y = (Math.random() - 0.5) * jitter;
      groupRef.current.rotation.z = (Math.random() - 0.5) * jitter * 0.3;
    }
  });

  return (
    <group ref={groupRef}>
      <Race radius={OUTER_RADIUS} tube={0.32} matchIntensity={bpfoIntensity} />
      <group ref={innerAssemblyRef}>
        <Race radius={INNER_RADIUS} tube={0.26} matchIntensity={bpfiIntensity} />
        <RollerRow zOffset={-ROW_OFFSET} />
        <RollerRow zOffset={ROW_OFFSET} />
      </group>
    </group>
  );
}

export interface BearingModel3DProps {
  bpfoIntensity?: number;
  bpfiIntensity?: number;
  severity?: number;
  autoRotate?: boolean;
  enableOrbitControls?: boolean;
  className?: string;
}

export default function BearingModel3D({
  bpfoIntensity = 0,
  bpfiIntensity = 0,
  severity = 0,
  autoRotate = true,
  enableOrbitControls = true,
  className,
}: BearingModel3DProps) {
  return (
    <div className={className}>
      <Canvas
        shadows={{ type: THREE.PCFShadowMap }}
        camera={{ position: [5, 3.2, 6], fov: 52 }}
        gl={{ antialias: true, alpha: true }}
      >
        <ambientLight intensity={0.18} />
        {/* key light */}
        <directionalLight
          position={[6, 8, 5]}
          intensity={1.6}
          color="#fff7ed"
          castShadow
        />
        {/* fill light */}
        <directionalLight position={[-6, 2, -3]} intensity={0.5} color="#7dd3fc" />
        {/* rim light */}
        <pointLight position={[0, -4, -6]} intensity={1.1} color="#22d3ee" />
        <Environment preset="city" environmentIntensity={0.35} />

        <BearingRig
          bpfoIntensity={bpfoIntensity}
          bpfiIntensity={bpfiIntensity}
          severity={severity}
          autoRotate={autoRotate}
        />

        {enableOrbitControls && (
          <OrbitControls
            enableDamping
            dampingFactor={0.08}
            minDistance={7}
            maxDistance={14}
            autoRotate={false}
          />
        )}
      </Canvas>
    </div>
  );
}
