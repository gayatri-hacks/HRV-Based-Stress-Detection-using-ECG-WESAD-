import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Waves } from "lucide-react";
import { useScanStore } from "@/store/useScanStore";

const scans = [
  {
    type: "ecg" as const,
    title: "ECG Scan",
    subtitle: "Electrocardiography",
    description: "Analyze a CSV ECG recording for a research stress prediction",
    icon: Waves,
    gradient: "from-accent to-primary",
  },
];

const ScanSelection = () => {
  const navigate = useNavigate();
  const setScanType = useScanStore((s) => s.setScanType);

  const handleSelect = (type: "ecg") => {
    setScanType(type);
    navigate(`/scan/${type}`);
  };

  return (
    <section className="py-20 px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-14"
      >
        <h1 className="text-4xl md:text-5xl font-extrabold font-display text-foreground tracking-tight mb-4">
          Analyze Your Health Signals
        </h1>
        <p className="text-lg text-muted-foreground max-w-xl mx-auto">
          Upload a genuine ECG CSV recording for a research stress prediction. This is not a medical device.
        </p>
      </motion.div>

      <div className="grid gap-6 max-w-xl mx-auto">
        {scans.map((scan, i) => (
          <motion.button
            key={scan.type}
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.15 * i }}
            whileHover={{ y: -6, scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => handleSelect(scan.type)}
            className="glass-card p-8 text-left group cursor-pointer transition-shadow hover:shadow-elevated"
            aria-label={`Select ${scan.title}`}
          >
            <div className={`inline-flex rounded-xl p-3 gradient-primary mb-5`}>
              <scan.icon className="h-7 w-7 text-primary-foreground" />
            </div>
            <h2 className="text-2xl font-bold font-display text-foreground mb-1">
              {scan.title}
            </h2>
            <p className="text-xs font-medium text-primary uppercase tracking-wider mb-3">
              {scan.subtitle}
            </p>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {scan.description}
            </p>
            <div className="mt-5 text-sm font-semibold text-primary group-hover:translate-x-1 transition-transform inline-flex items-center gap-1">
              Get Started →
            </div>
          </motion.button>
        ))}
      </div>
    </section>
  );
};

export default ScanSelection;
