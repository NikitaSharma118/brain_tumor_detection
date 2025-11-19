import {Card } from "../assets";
import styles, { layout } from '../style';

const content = () => (
    <section id="cta" className={layout.sectionReverse}>
        <div className={layout.sectionImgReverse}>
            <img src={Card} alt="card" className="w-[600px] h-[600px] object-contain relative z-[5]"/>
        </div>

        <div className={layout.sectionInfo}>
            <h2 className={`${styles.heading2} ml-10`}>Our Simple 3-Step Process</h2>
            <p className={`${styles.paragraph} max-w-[470px] ml-10 mt-4`}>Get results in minutes, not weeks. Our intuitive platform guides you from file upload to final analysis.</p>
            <ul className="ml-10 mt-8 space-y-6">
                <li>
                    <span className="text-[#00C6FF] font-semibold text-[20px]">
                        1. Upload Securely
                    </span>
                    <p className="text-dimWhite text-[16px] leading-[26px] max-w-[460px]">
                        Drag and drop your JPG, JPEG, or PNG brain scan files into our encrypted portal.
                    </p>
                </li>
                <li>
                    <span className="text-[#00C6FF] font-semibold text-[20px]">
                        2. AI Analysis
                    </span>
                    <p className="text-dimWhite text-[16px] leading-[26px] max-w-[460px]">
                        Our proprietary neural network rapidly processes the images, identifying potential tumor regions with high precision. 
                   </p>
                </li>
                <li>
                    <span className="text-[#00C6FF] font-semibold text-[20px]">
                        3. View Instant Insights
                    </span>
                    <p className="text-dimWhite text-[16px] leading-[26px] max-w-[460px]">
                        Access your personalized visual report showing annotated areas and detection metrics.
                    </p>
                </li>
            </ul>
        </div>
    </section>
)

export default content