import { BrowserRouter, Route, Routes } from "react-router-dom";
import { AppShell } from "@/components/app-shell";
import { ScanMobileShell } from "@/components/scan-mobile-shell";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/scan" element={<ScanMobileShell />} />
        <Route path="/*" element={<AppShell />} />
      </Routes>
    </BrowserRouter>
  );
}
