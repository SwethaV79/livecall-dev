import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useIsAuthenticated, useMsal } from '@azure/msal-react';
import {
    BarChart3,
    Shield,
    Zap,
    Users,
    TrendingUp,
    Phone,
    Brain,
    CheckCircle,
    ArrowRight,
    Play,
    Mic,
    Activity,
    Clock,
    Target,
    Award
} from 'lucide-react';
import FloatingFooter from '@/components/common/FloatingFooter';

const HomePage = () => {
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const isAuthenticated = useIsAuthenticated();
    const navigate = useNavigate();
    const { instance } = useMsal();
    const [scrollDirection, setScrollDirection] = useState('down');
    const [lastScrollY, setLastScrollY] = useState(0);
    const [animationPhase, setAnimationPhase] = useState(0);

    // Handle login
    const handleLogin = async () => {
        try {
            setIsLoading(true);
            setError('');
            const loginResponse = await instance.loginPopup({
                scopes: ["User.Read"],
                prompt: "select_account"
            });

            if (loginResponse) {
                navigate('/index', { replace: true });
            }
        } catch (error) {
            console.error("Login failed:", error);
            setError('Login failed. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    // Scroll direction detection and animation control
    useEffect(() => {
        const handleScroll = () => {
            const currentScrollY = window.scrollY;
            const scrollDelta = Math.abs(currentScrollY - lastScrollY);

            if (currentScrollY > lastScrollY && scrollDelta > 5) {
                setScrollDirection('down');
                setAnimationPhase((prev) => (prev + 1) % 3);
            } else if (currentScrollY < lastScrollY && scrollDelta > 5) {
                setScrollDirection('up');
                setAnimationPhase((prev) => (prev + 1) % 3);
            }
            setLastScrollY(currentScrollY);
        };

        window.addEventListener('scroll', handleScroll, { passive: true });
        return () => window.removeEventListener('scroll', handleScroll);
    }, [lastScrollY]);

    // Auto animation cycle
    useEffect(() => {
        const interval = setInterval(() => {
            setAnimationPhase((prev) => (prev + 1) % 3);
        }, 3000);

        return () => clearInterval(interval);
    }, []);

    // Animated counter hook
    const useCounter = (end, duration = 2000) => {
        const [count, setCount] = useState(0);

        useEffect(() => {
            let startTime;
            const animate = (currentTime) => {
                if (!startTime) startTime = currentTime;
                const progress = Math.min((currentTime - startTime) / duration, 1);
                setCount(Math.floor(progress * end));
                if (progress < 1) {
                    requestAnimationFrame(animate);
                }
            };
            requestAnimationFrame(animate);
        }, [end, duration]);

        return count;
    };

    const satisfactionRate = useCounter(94);
    const responseTime = useCounter(73);
    const efficiency = useCounter(86);

    const features = [
        {
            icon: <Brain className="w-8 h-8 text-blue-700" />,
            title: "AI-Powered Analysis",
            description: "Real-time emotion detection and sentiment analysis during live calls",
            color: "bg-white border-gray-200 hover:border-blue-300"
        },
        {
            icon: <Activity className="w-8 h-8 text-blue-700" />,
            title: "Live Monitoring",
            description: "Monitor call progress, agent performance, and customer satisfaction in real-time",
            color: "bg-white border-gray-200 hover:border-blue-300"
        },
        {
            icon: <Shield className="w-8 h-8 text-blue-700" />,
            title: "Risk Assessment",
            description: "Automatic escalation risk detection and proactive intervention suggestions",
            color: "bg-white border-gray-200 hover:border-blue-300"
        },
        {
            icon: <BarChart3 className="w-8 h-8 text-blue-700" />,
            title: "Advanced Analytics",
            description: "Comprehensive reporting and insights to improve customer service quality",
            color: "bg-white border-gray-200 hover:border-blue-300"
        }
    ];

    const benefits = [
        "Reduce call resolution time by 40%",
        "Improve customer satisfaction scores",
        "Enhance agent performance tracking",
        "Automate quality assurance processes",
        "Prevent customer escalations proactively"
    ];

    const stats = [
        {
            icon: <Award className="w-8 h-8 text-blue-700" />,
            value: satisfactionRate,
            label: "Customer Satisfaction",
            suffix: "%",
            color: "text-blue-700",
            bgColor: "bg-white border-gray-200"
        },
        {
            icon: <Clock className="w-8 h-8 text-blue-700" />,
            value: responseTime,
            label: "Faster Response Time",
            suffix: "%",
            color: "text-blue-700",
            bgColor: "bg-white border-gray-200"
        },
        {
            icon: <Target className="w-8 h-8 text-blue-700" />,
            value: efficiency,
            label: "Critical Issues Resolved",
            suffix: "%",
            color: "text-blue-700",
            bgColor: "bg-white border-gray-200"
        }
    ];

    const getCardTransform = (index) => {
        const baseTransforms = [
            'translate-x-0 translate-y-0',
            'translate-x-16 translate-y-8',
            'translate-x-32 translate-y-16'
        ];

        const slideInTransforms = [
            'translate-x-0 translate-y-0',
            'translate-x-8 translate-y-4',
            'translate-x-16 translate-y-8'
        ];

        if (animationPhase === 0) {
            return baseTransforms[index];
        } else if (animationPhase === 1) {
            return slideInTransforms[index];
        } else {
            return baseTransforms[(index + 1) % 3];
        }
    };

    const getCardOpacity = (index) => {
        if (animationPhase === index) return 'opacity-100';
        return 'opacity-80';
    };

    const getCardZIndex = (index) => {
        if (animationPhase === index) return 'z-30';
        if (animationPhase === (index + 1) % 3) return 'z-20';
        return 'z-10';
    };

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Subtle background pattern */}
            <div className="absolute inset-0 opacity-3">
                <div className="absolute top-0 left-0 w-full h-full" style={{
                    backgroundImage: `radial-gradient(circle at 2px 2px, #e5e7eb 1px, transparent 0)`,
                    backgroundSize: '40px 40px'
                }}></div>
            </div>

            {/* Navigation */}
            <nav className="relative z-10 bg-white border-b border-gray-200 shadow-sm">
                <div className="mx-auto px-4 sm:px-6 lg:px-8 py-3 sm:py-4 flex justify-between items-center">
                    <div className="flex items-center space-x-3">
                        <img
                            src="/logo.png"
                            alt="Company Logo"
                            className="h-8 w-auto sm:h-10 md:h-12 lg:h-14 xl:h-16 object-contain"
                        />
                    </div>
                    <div className="flex items-center space-x-2 sm:space-x-3">
                        <div className="bg-blue-700 p-1.5 sm:p-2 rounded-lg shadow-sm">
                            <Phone className="w-4 h-4 sm:w-5 sm:h-5 lg:w-6 lg:h-6 text-white" />
                        </div>
                        <span className="text-lg sm:text-xl lg:text-2xl font-semibold text-gray-900">
                            Call Insight
                        </span>
                    </div>
                </div>
            </nav>

            {/* Hero Section */}
            <div className="relative z-10 mx-auto px-4 sm:px-6 lg:px-8 pt-12 sm:pt-16 pb-16 sm:pb-20">



                {/* Blurred Background Image */}
                {/* Lightly Blurred Background Image */}
                <div className="absolute top-0 left-0 w-full h-full max-h-[700px] -z-10 overflow-hidden">
                    <img
                        src="/hero-bg.jpg"
                        alt="Background"
                        className="w-full h-full object-cover [filter:blur(2px)]"
                    />
                    <div className="absolute inset-0 bg-gradient-to-b from-white/30 to-white"></div>
                </div>


                <div className="text-center mb-16 sm:mb-20">
                    <div className="inline-flex items-center space-x-2 bg-blue-50 text-blue-800 rounded-full px-3 sm:px-4 py-2 mb-6 sm:mb-8 border border-blue-200">
                        <Zap className="w-4 h-4" />
                        <span className="text-sm font-medium">AI-Powered Customer Support</span>
                    </div>

                    <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl xl:text-7xl font-bold mb-4 sm:mb-6 text-gray-900 leading-tight">
                        Transform Your
                        <span className="block text-blue-700">
                            Customer Support
                        </span>
                    </h1>

                    <p className="text-lg sm:text-xl lg:text-2xl text-gray-600 mb-8 sm:mb-12 max-w-3xl mx-auto leading-relaxed">
                        Real-time AI analysis, sentiment detection, and performance insights
                        that revolutionize how you handle customer interactions.
                    </p>

                    <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12 sm:mb-16">
                        <button
                            onClick={handleLogin}
                            disabled={isLoading}
                            className="bg-blue-700 hover:bg-blue-800 text-white px-6 sm:px-8 py-3 sm:py-4 rounded-lg font-semibold text-base sm:text-lg transition-all duration-300 transform hover:scale-105 disabled:opacity-70 disabled:cursor-not-allowed flex items-center space-x-2 sm:space-x-3 shadow-md"
                        >
                            {isLoading ? (
                                <>
                                    <div className="animate-spin rounded-full h-4 w-4 sm:h-5 sm:w-5 border-2 border-white border-t-transparent"></div>
                                    <span>Signing In...</span>
                                </>
                            ) : (
                                <>
                                    <span>Get Started</span>
                                    <ArrowRight className="w-4 h-4 sm:w-5 sm:h-5" />
                                </>
                            )}
                        </button>
                    </div>

                    {error && (
                        <div className="bg-red-50 border border-red-200 text-red-700 px-4 sm:px-6 py-3 rounded-lg max-w-md mx-auto">
                            {error}
                        </div>
                    )}
                </div>

                {/* Dashboard Preview Section */}
                <div className="mb-16 sm:mb-20">
                    <div className="text-center mb-8 sm:mb-12">
                        <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 mb-4">Live Dashboard Preview</h2>
                        <p className="text-base sm:text-lg text-gray-600">See how our AI monitors and analyzes customer interactions in real-time</p>
                    </div>

                    <div className="bg-white rounded-lg sm:rounded-2xl p-3 sm:p-4 md:p-8 shadow-lg border border-gray-200">
                        <div className="text-center text-gray-600">
                            <div className="bg-gray-50 rounded-lg sm:rounded-xl p-3 sm:p-4 md:p-8 border-2 border-dashed border-gray-300 overflow-hidden">
                                {/* Responsive Container */}
                                <div className="relative w-full mx-auto 
                                    max-w-2xl h-48
                                    md:max-w-4xl md:h-80
                                    lg:max-w-5xl lg:h-96
                                    xl:max-w-6xl xl:h-[500px]">
                                    {/* Card 1 */}
                                    <div className={`absolute inset-0 transition-all duration-1000 ease-in-out transform ${getCardTransform(0)} ${getCardOpacity(0)} ${getCardZIndex(0)}`}>
                                        <div className="w-full h-full bg-white rounded-lg sm:rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                                            <img
                                                src="/img1.png"
                                                alt="Dashboard Interface"
                                                className="w-full h-full lg:object-contain object-cover transition-all duration-500"
                                                style={{
                                                    maxWidth: '100%',
                                                    maxHeight: '100%',
                                                }}
                                                onError={(e) => {
                                                    e.currentTarget.style.display = 'none';
                                                    const fallback = e.currentTarget.nextElementSibling as HTMLElement;
                                                    if (fallback) fallback.style.display = 'flex';
                                                }}
                                            />
                                            <div className="hidden w-full h-full bg-gray-100 rounded-lg sm:rounded-xl flex items-center justify-center">
                                                <div className="text-center">
                                                    <Activity className="w-12 sm:w-16 h-12 sm:h-16 text-blue-700 mx-auto mb-4" />
                                                    <p className="text-base sm:text-lg font-semibold text-gray-800">Live Analytics Dashboard</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Card 2 */}
                                    <div
                                        className={`absolute inset-0 transition-all duration-1000 ease-in-out transform ${getCardTransform(1)} ${getCardOpacity(1)} ${getCardZIndex(1)}`}
                                    >
                                        <div className="w-full h-full bg-white rounded-lg sm:rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                                            <img
                                                src="/img2.png"
                                                alt="AI Insights Dashboard"
                                                className="w-full h-full lg:object-contain object-cover transition-all duration-500"
                                                onError={(e) => {
                                                    e.currentTarget.style.display = 'none';
                                                    (e.currentTarget.nextElementSibling as HTMLElement).style.display = 'flex';
                                                }}
                                            />
                                            <div className="hidden w-full h-full bg-gray-100 rounded-lg sm:rounded-xl flex items-center justify-center">
                                                <div className="text-center">
                                                    <Brain className="w-12 sm:w-16 h-12 sm:h-16 text-blue-700 mx-auto mb-4" />
                                                    <p className="text-base sm:text-lg font-semibold text-gray-800">AI-Powered Insights</p>
                                                    <p className="text-sm text-gray-600 mt-2">Intelligent sentiment analysis</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Card 3 */}
                                    <div
                                        className={`absolute inset-0 transition-all duration-1000 ease-in-out transform ${getCardTransform(2)} ${getCardOpacity(2)} ${getCardZIndex(2)}`}
                                    >
                                        <div className="w-full h-full bg-white rounded-lg sm:rounded-xl shadow-lg border border-gray-200 overflow-hidden">
                                            <img
                                                src="/img3.png"
                                                alt="Performance Metrics Dashboard"
                                                className="w-full h-full lg:object-contain object-cover transition-all duration-500"
                                                onError={(e) => {
                                                    e.currentTarget.style.display = 'none';
                                                    (e.currentTarget.nextElementSibling as HTMLElement).style.display = 'flex';
                                                }}
                                            />
                                            <div className="hidden w-full h-full bg-gray-100 rounded-lg sm:rounded-xl flex items-center justify-center">
                                                <div className="text-center">
                                                    <Target className="w-12 sm:w-16 h-12 sm:h-16 text-blue-700 mx-auto mb-4" />
                                                    <p className="text-base sm:text-lg font-semibold text-gray-800">Performance Metrics</p>
                                                    <p className="text-sm text-gray-600 mt-2">Advanced reporting & KPIs</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div className="mt-4 sm:mt-6 text-sm text-gray-500">
                                    <div className="flex justify-center space-x-2">
                                        <div className={`w-2 h-2 sm:w-3 sm:h-3 rounded-full transition-all duration-300 ${animationPhase === 0 ? 'bg-blue-600 scale-110' : 'bg-gray-300'}`}></div>
                                        <div className={`w-2 h-2 sm:w-3 sm:h-3 rounded-full transition-all duration-300 ${animationPhase === 1 ? 'bg-blue-600 scale-110' : 'bg-gray-300'}`}></div>
                                        <div className={`w-2 h-2 sm:w-3 sm:h-3 rounded-full transition-all duration-300 ${animationPhase === 2 ? 'bg-blue-600 scale-110' : 'bg-gray-300'}`}></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Stats Section */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 sm:gap-8 mb-16 sm:mb-20">
                    {stats.map((stat, index) => (
                        <div key={index} className={`${stat.bgColor} rounded-lg sm:rounded-2xl p-6 sm:p-8 border-2 text-center hover:shadow-lg transition-all duration-300 transform hover:scale-105`}>
                            <div className="flex justify-center mb-4">
                                {stat.icon}
                            </div>
                            <div className={`text-3xl sm:text-4xl font-bold mb-2 ${stat.color}`}>
                                {stat.value}{stat.suffix}
                            </div>
                            <div className="text-gray-700 font-medium text-sm sm:text-base">{stat.label}</div>
                        </div>
                    ))}
                </div>

                {/* Features Grid */}
                <div className="mb-16 sm:mb-20">
                    <div className="text-center mb-8 sm:mb-12">
                        <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 mb-4">Powerful Features</h2>
                        <p className="text-base sm:text-lg text-gray-600">Everything you need to deliver exceptional customer support</p>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 sm:gap-8">
                        {features.map((feature, index) => (
                            <div
                                key={index}
                                className={`${feature.color} rounded-lg sm:rounded-2xl p-4 sm:p-6 border-2 hover:shadow-lg transition-all duration-300 transform hover:scale-105`}
                                style={{ animationDelay: `${index * 0.1}s` }}
                            >
                                <div className="mb-4">{feature.icon}</div>
                                <h3 className="text-lg sm:text-xl font-semibold mb-3 text-gray-900">{feature.title}</h3>
                                <p className="text-gray-700 text-sm leading-relaxed">{feature.description}</p>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Benefits Section */}
                <div className="bg-white rounded-lg sm:rounded-2xl p-6 sm:p-8 border-2 border-gray-200 mb-16 sm:mb-20 shadow-sm">
                    <h2 className="text-2xl sm:text-3xl font-bold text-center mb-8 sm:mb-12 text-gray-900">Why Choose Call Insight?</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
                        {benefits.map((benefit, index) => (
                            <div key={index} className="flex items-center space-x-3">
                                <CheckCircle className="w-5 h-5 sm:w-6 sm:h-6 text-blue-700 flex-shrink-0" />
                                <span className="text-gray-800 font-medium text-sm sm:text-base">{benefit}</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* CTA Section */}
                <div className="text-center bg-white rounded-lg sm:rounded-2xl p-8 sm:p-12 shadow-lg border border-gray-200">
                    <h2 className="text-3xl sm:text-4xl font-bold mb-4 sm:mb-6 text-gray-900">Ready to Transform Your Support?</h2>
                    <p className="text-lg sm:text-xl text-gray-600 mb-6 sm:mb-8 max-w-2xl mx-auto">
                        Join leading companies using Call Insight to deliver exceptional customer experiences.
                    </p>
                    <button
                        onClick={handleLogin}
                        disabled={isLoading}
                        className="bg-blue-700 hover:bg-blue-800 text-white px-8 sm:px-12 py-3 sm:py-4 rounded-lg sm:rounded-xl font-semibold text-lg sm:text-xl transition-all duration-300 transform hover:scale-105 disabled:opacity-70 disabled:cursor-not-allowed flex items-center space-x-2 sm:space-x-3 mx-auto shadow-lg"
                    >
                        {isLoading ? (
                            <>
                                <div className="animate-spin rounded-full h-5 w-5 sm:h-6 sm:w-6 border-2 border-white border-t-transparent"></div>
                                <span>Signing In...</span>
                            </>
                        ) : (
                            <>
                                <span>Get Started Today</span>
                                <TrendingUp className="w-5 h-5 sm:w-6 sm:h-6" />
                            </>
                        )}
                    </button>
                </div>
            </div>

            {/* Footer */}
            <footer className="bg-gray-900 text-white py-8 sm:py-12 mt-16 sm:mt-20">
                <div className="mx-auto px-4 sm:px-6 lg:px-8 text-center">
                    <div className="flex items-center justify-center space-x-2 sm:space-x-3 mb-4">
                        <div className="bg-blue-700 p-1.5 sm:p-2 rounded-lg">
                            <Phone className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
                        </div>
                        <span className="text-xl sm:text-2xl font-bold">Call Insight</span>
                    </div>
                    <p className="text-gray-400 text-sm sm:text-base">Transforming customer support with AI-powered insights</p>
                </div>
            </footer>
            <FloatingFooter />
        </div>
    );
};

export default HomePage;