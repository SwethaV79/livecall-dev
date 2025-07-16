import { Configuration, PublicClientApplication, BrowserCacheLocation } from "@azure/msal-browser";

// MSAL configuration
const msalConfig: Configuration = {
  auth: {
    clientId: "9297e893-98da-423c-8c1f-24c626c6c47a",
    authority: "https://login.microsoftonline.com/f5791d91-daca-4d28-8700-680f7a2f8b6a",
    redirectUri: window.location.origin,
    postLogoutRedirectUri: window.location.origin,
    navigateToLoginRequestUrl: true
  },
  cache: {
    cacheLocation: BrowserCacheLocation.LocalStorage,
    storeAuthStateInCookie: true
  }
};

// Login request configuration
export const loginRequest = {
  scopes: ["User.Read", "openid", "profile", "email"]
};

// Initialize MSAL instance
export const msalInstance = new PublicClientApplication(msalConfig);

// Handle redirect after login
msalInstance.initialize().then(() => {
  msalInstance.handleRedirectPromise().then((response) => {
    if (response) {
      const accounts = msalInstance.getAllAccounts();
      if (accounts.length > 0) {
        msalInstance.setActiveAccount(accounts[0]);
        // Remove this line:
        // window.location.replace('/index');
      }
    }
  }).catch(error => {
    console.error("Redirect error:", error);
  });
});

// Single login function implementation
export const handleMsalLogin = async () => {
  try {
    await msalInstance.loginRedirect({
      ...loginRequest,
      prompt: 'select_account'
    });
  } catch (error) {
    console.error("Login failed:", error);
    throw error;
  }
};

