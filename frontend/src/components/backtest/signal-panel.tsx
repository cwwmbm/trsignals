import type { SignalInfo } from "@/api";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type SignalPanelProps = {
  signalKind: "single" | "combined";
  onSignalKindChange: (kind: "single" | "combined") => void;
  signal: string;
  onSignalChange: (signal: string) => void;
  signalA: string;
  onSignalAChange: (signal: string) => void;
  signalB: string;
  onSignalBChange: (signal: string) => void;
  comboMode: "and" | "or";
  onComboModeChange: (mode: "and" | "or") => void;
  signalOptions: SignalInfo[];
};

function SignalSelect({
  id,
  label,
  value,
  onChange,
  options,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: SignalInfo[];
}) {
  return (
    <div className="flex flex-col gap-2">
      <Label htmlFor={id}>{label}</Label>
      <Select value={value} onValueChange={(v) => v && onChange(v)}>
        <SelectTrigger id={id} className="w-full font-mono">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((item) => (
            <SelectItem key={item.name} value={item.name} className="font-mono">
              {item.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

export function SignalPanel({
  signalKind,
  onSignalKindChange,
  signal,
  onSignalChange,
  signalA,
  onSignalAChange,
  signalB,
  onSignalBChange,
  comboMode,
  onComboModeChange,
  signalOptions,
}: SignalPanelProps) {
  return (
    <div className="mt-5 rounded-lg border border-border/60 bg-muted/30 p-5">
      <h3 className="text-sm font-semibold">Signal</h3>
      <RadioGroup
        value={signalKind}
        onValueChange={(v) => onSignalKindChange(v as "single" | "combined")}
        className="mt-3 flex flex-row gap-6"
      >
        <div className="flex items-center gap-2">
          <RadioGroupItem value="single" id="r-single" />
          <Label htmlFor="r-single" className="font-normal">
            Single
          </Label>
        </div>
        <div className="flex items-center gap-2">
          <RadioGroupItem value="combined" id="r-combined" />
          <Label htmlFor="r-combined" className="font-normal">
            Combined
          </Label>
        </div>
      </RadioGroup>

      {signalKind === "single" ? (
        <div className="mt-4 max-w-md">
          <SignalSelect
            id="signal"
            label="Signal"
            value={signal}
            onChange={onSignalChange}
            options={signalOptions}
          />
        </div>
      ) : (
        <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
          <SignalSelect
            id="signal-a"
            label="Primary"
            value={signalA}
            onChange={onSignalAChange}
            options={signalOptions}
          />
          <div className="flex flex-col gap-2">
            <Label htmlFor="combo-mode">Mode</Label>
            <Select value={comboMode} onValueChange={(v) => v && onComboModeChange(v as "and" | "or")}>
              <SelectTrigger id="combo-mode" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="and">AND</SelectItem>
                <SelectItem value="or">OR</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <SignalSelect
            id="signal-b"
            label="Secondary"
            value={signalB}
            onChange={onSignalBChange}
            options={signalOptions}
          />
        </div>
      )}
    </div>
  );
}
