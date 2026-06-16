import { memo, useMemo } from "react";
import type { IndicatorInfo } from "@/api";
import { groupIndicators } from "@/lib/indicator-groups";
import { cn } from "@/lib/utils";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

function IndicatorSelectInner({
  id,
  value,
  onChange,
  indicators,
  className,
  placeholder = "Select indicator",
  compact = false,
}: {
  id?: string;
  value: string;
  onChange: (value: string) => void;
  indicators: IndicatorInfo[];
  className?: string;
  placeholder?: string;
  compact?: boolean;
}) {
  const groups = useMemo(() => groupIndicators(indicators), [indicators]);
  const selected = indicators.find((item) => item.id === value);

  return (
    <Select value={value || undefined} onValueChange={(v) => v && onChange(v)}>
      <SelectTrigger
        id={id}
        size={compact ? "sm" : "default"}
        className={cn(compact && "h-8 text-xs", className)}
      >
        <SelectValue placeholder={placeholder}>
          {() => selected?.label ?? placeholder}
        </SelectValue>
      </SelectTrigger>
      <SelectContent
        className={cn(
          compact &&
            "max-h-[min(28rem,70vh)] min-w-[var(--anchor-width)] w-max max-w-[min(28rem,90vw)] [&_[data-slot=select-label]]:py-0.5 [&_[data-slot=select-label]]:text-[10px] [&_[data-slot=select-item]]:py-0.5 [&_[data-slot=select-item]]:pr-7 [&_[data-slot=select-item]]:text-xs",
          !compact && "max-h-72",
        )}
      >
        {[...groups.entries()].map(([category, items]) => (
          <SelectGroup key={category}>
            <SelectLabel>{category}</SelectLabel>
            {items.map((item) => (
              <SelectItem key={item.id} value={item.id} className={cn(compact && "text-xs")}>
                {item.label}
              </SelectItem>
            ))}
          </SelectGroup>
        ))}
      </SelectContent>
    </Select>
  );
}

export const IndicatorSelect = memo(IndicatorSelectInner);

export function indicatorLabel(indicators: IndicatorInfo[], value: string) {
  return indicators.find((item) => item.id === value)?.label ?? value;
}
