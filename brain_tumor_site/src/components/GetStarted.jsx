//ABOUT US SECTION

import Homepage from "../assets";
import styles, {layout} from "../style";

const GetStarted = () => (
    <section id="about" className="py-20 bg-[#0A0C16]">
        <div className="max-w-7xl mx-auto px-6 md:px-12 flex flex-col md:flex-row items-center">
            {/* Left Side: Image */}
            <div className="w-full md:w-1/2 flex justify-center p-8">
                <img
                    src={Homepage}
                    alt="homepage"
                    className="w-full max-w-sm rounded-xl shadow-2xl shadow-sky-900/50"
                />
            </div>
            {/* Right Side: Content */}
            <div className="w-full md:w-1/2 mt-12 md:mt-0 md:pl-16">
                <h2 className="text-4xl font-extrabold text-white mb-4">Built by Experts. Backed by Data.</h2>
                <p className="text-gray-400 text-lg mb-6">
                    MindAI was founded for the dedicated team of clinical radiologists, data scientists, and AI engineers committed to accelerating accurate diagnosis. We believe that integrating advanced technology with medical expertise leads to better patient outcomes.
                </p>
                <p className="text-gray-400 text-lg mb-6">
                    Our platform is continuously validated against large, diverse datasets and designed to seamlessly assist medical professionals in identifying potential abnormalities earlier and with greater confidence.
                </p>
            </div>
        </div>
    </section>
);

export default GetStarted