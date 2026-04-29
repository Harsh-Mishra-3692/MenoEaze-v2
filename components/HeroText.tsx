"use client";

import { motion } from "framer-motion";

const sentence = {
    hidden: { opacity: 1 },
    visible: {
        opacity: 1,
        transition: {
            staggerChildren: 0.04,
        },
    },
};

const letter = {
    hidden: { opacity: 0, y: 40 },
    visible: {
        opacity: 1,
        y: 0,
        transition: {
            duration: 0.6,
            ease: [0.22, 1, 0.36, 1] as const,
        },
    },
};

export default function HeroText() {
    const text = "Track. Understand. Heal.";

    return (
        <motion.h1
            variants={sentence}
            initial="hidden"
            animate="visible"
            className="text-5xl md:text-7xl font-bold text-center leading-tight"
        >
            {text.split("").map((char, index) => (
                <motion.span key={index} variants={letter}>
                    {char === " " ? "\u00A0" : char}
                </motion.span>
            ))}
        </motion.h1>
    );
}