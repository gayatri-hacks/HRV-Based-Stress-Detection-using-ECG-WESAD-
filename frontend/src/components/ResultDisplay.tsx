import { motion } from "framer-motion";
import ResultSlider from "./ResultSlider";
import PDFExportButton from "./PDFExportButton";
import { useScanStore } from "@/store/useScanStore";

const ResultDisplay = () => {
  const result = useScanStore((s) => s.result);
  const scanType = useScanStore((s) => s.scanType);

  if (!result) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
      className="space-y-6"
    >
      <div className="text-center">
        <h2 className="text-2xl font-bold font-display text-foreground mb-1">
          Analysis Complete
        </h2>
        <p className="text-sm text-muted-foreground">
          {scanType?.toUpperCase()} scan results •{" "}
          {new Date(result.timestamp).toLocaleString()}
        </p>
      </div>

      <div className="max-w-sm mx-auto">
        <ResultSlider
          label="Stress Level"
          value={result.stressLevel}
          confidence={result.stressConfidence}
          delay={200}
        />
      </div>

      <div className="flex justify-center pt-2">
        <PDFExportButton />
      </div>
    </motion.div>
  );
};

export default ResultDisplay;
