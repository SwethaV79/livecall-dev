import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { MsalProvider, useIsAuthenticated, useMsal } from "@azure/msal-react";
import { msalInstance } from "./config/msal-config";
import HomePage from "./pages/home";
import Index from "./pages/Index";

const queryClient = new QueryClient();

// Protected Route Component
const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const isAuthenticated = useIsAuthenticated();
  const { accounts } = useMsal();

  // Wait for MSAL to finish loading accounts
  if (isAuthenticated === undefined || (accounts && accounts.length === 0 && isAuthenticated === false)) {
    // Show nothing or a loading spinner while determining auth state
    return null;
  }

  return isAuthenticated ? <>{children}</> : <Navigate to="/" replace />;
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <MsalProvider instance={msalInstance}>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route
              path="/index"
              element={
                <ProtectedRoute>
                  <Index />
                </ProtectedRoute>
              }
            />
          </Routes>
        </BrowserRouter>
      </MsalProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

// export default App;
//     </TooltipProvider>
//   </QueryClientProvider>
// );

export default App;
